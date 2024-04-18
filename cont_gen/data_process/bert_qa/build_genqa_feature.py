"""
Pipelines to build generative QA features.

    Text process
    1. split_paragraphs: split doc into paragraphs.
    2. remove_space_keep_mapping: remove extra spaces

    Get paragraph examples on processed text
    3. get_genqa_examples
    4. tokenize each paragraph

    Tokenization:
    3. tokenize_paragraph: tokenize each paragraph
    
    Build Features:
        The ideal case is that, for each paragraph, ask all questions.
        To improve efficiency of training, questions with null answers of each paragraph should be sampled.

    4. Given original data, find the 
"""

from pathlib import Path
import re
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from typing import List, Tuple, Dict, Optional, Union

from transformers import PreTrainedTokenizerFast, AutoTokenizer

from cont_gen.data_process.basic import NaturalParagraph
from cont_gen.data_process.utils import (
    cut_spans_return_offset, 
    remove_spans_and_return_mapping,
    merge_spans,
    span_contain,
    relocate_spans_into_range,
    reverse_char_map,
    group_by
)

def split_paragraphs(doc):
    """
    Split natural paragraphs.

    Return
        natural_paragraphs: a list, each element is a paragraph dict:
            p_text (str): paragraph text
            offset (str): start position in original doc
            length (str): length of p_text
    """
    # get linebreak spans
    break_spans = [] # list of tuple: (start, end)
    pat = r'[\n]+'
    for m in re.finditer(pat, doc):
        break_spans.append(m.span())
    
    text_parts, offsets = cut_spans_return_offset(doc, break_spans)
    
    # get natural paragraphs
    nat_paras = [
        {
            'p_text': txt,
            'offset': ofs,
            'length': len(txt)
        } for txt, ofs in zip(text_parts, offsets)
    ]
    
    return nat_paras

def remove_extra_space(ori_text):
    """
    Substitute >=2 space with one space and return the char mapping.

    Return:
        new_text (str)
        new_to_old_mapping (List[int]): char mapping from new_text to ori_text
    """
    pat = r'[ ]{2,}'
    break_spans = [m.span() for m in re.finditer(pat, ori_text)]
    text_parts, offsets = cut_spans_return_offset(ori_text, break_spans)

    new_to_old_mapping = []
    for t, ofs in zip(text_parts, offsets):
        new_to_old_mapping.extend([ofs + i for i in range(len(t))])
        # Add the pos for the added space
        new_to_old_mapping.append(new_to_old_mapping[-1] + 1)
    
    if len(new_to_old_mapping) > 0:
        _ = new_to_old_mapping.pop(-1)
    new_text = ' '.join(text_parts)
    return new_text, new_to_old_mapping


def get_genqa_examples(dataset):
    """
    Return qa examples. Each example is a paragraph with valid questions 
    and answer spans.

    Return: a list of examples.
        contract_index
        title
        offset: offset in raw contract
        length: original length before removing extra spaces
        clean_text: text removing extra spaces
        char_map: map from clean text to ori text, in the paragraph range.
        qas: a list of question and answer spans
            clause_id
            clause_name
            answer_spans: List of spans (Tuple[int, int])
    """
    examples = []
    for ctt_i, data in tqdm(list(enumerate(dataset)), ncols = 80):
        # handle each contract
        title = data['title']
        doc_data = data['paragraphs'][0]
        doc = doc_data['context']

        # split paragraphs
        nat_paras = split_paragraphs(doc)

        # get spans for each clause
        cla2spans = {}
        cla_names = [qas['id'].split('__')[1] for qas in doc_data['qas']]
        for cla_i, qas in enumerate(doc_data['qas']):
                # check each clause
                spans = [[k['answer_start'], k['answer_start'] + len(k['text'])] 
                            for k in qas['answers']]
                cla2spans[cla_i] = merge_spans(spans)

        # handle each paragraph
        for pi, nat_para in enumerate(nat_paras):
            offset = nat_para['offset']
            length = nat_para['length']

            clean_text, char_map = remove_extra_space(nat_para['p_text'])
            old2new_map = reverse_char_map(char_map)

            # find spans that fall into the paragraph and relocate the position
            # relative to the paragraph
            para_cla2spans = {
                cla_i: list(relocate_spans_into_range(spans, offset, offset + length))
                    for cla_i, spans in cla2spans.items()
            }
            qas = []
            for cla_i, spans in para_cla2spans.items():
                if len(spans) == 0:
                    continue

                clean_spans = [[old2new_map[span[0]], old2new_map[span[1]-1] + 1] for span in spans]
                qas.append({
                    'clause_id': cla_i,
                    'clause_name': cla_names[cla_i],
                    'reloc_spans': spans,
                    'answer_spans': clean_spans
                })
            example = {
                'contract_index': ctt_i,
                'para_index': pi,
                'title': title,
                'offset': offset,
                'length': length,
                'clean_text': clean_text,
                'char_map': char_map,
                'qas': qas
            }
            examples.append(example)
    
    return examples

def fast_tokenize(text, tokenizer):
    enc = tokenizer(
        text, truncation = False, add_special_tokens = False, verbose = False
    )
    doc_token_ids = enc.input_ids
    token_to_char: List[Tuple[int, int]] = [
        list(enc.token_to_chars(i)) 
            for i in range(len(doc_token_ids))
    ] # the end not included
    return doc_token_ids, token_to_char

def tokenize_genqa_examples(examples, tokenizer, num_cpu = 1):
    all_clean_text = [k['clean_text'] for k in examples]
    _func = partial(fast_tokenize, tokenizer = tokenizer)
    results = []
    with Pool(num_cpu) as p:
        bar = tqdm(total = len(all_clean_text), desc = 'Para Tokenization')
        for r in p.imap(_func, all_clean_text, chunksize = 100):
            results.append(r)
            bar.update()
        bar.close()
    
    all_save_r = []
    for i in range(len(examples)):
        save_r = {
            'contract_index': examples[i]['contract_index'],
            'para_index': examples[i]['para_index'],
            'token_ids': results[i][0],
            'token_to_char': results[i][1]
        }
        all_save_r.append(save_r)

    return all_save_r

def build_genqa_features_for_contract(examples, cla_i, q_text):
    features = []
    for exa in examples:
        targ_qa = [qa for qa in exa['qas'] if qa['clause_id'] == cla_i]
        if len(targ_qa) == 0:
            answer_spans = []
            no_answer = True
            # input = exa['clean_text'] + q_text + 'No such clause'
        else:
            answer_spans = targ_qa[0]['answer_spans']
            no_answer = False
        
            # input = (exa['clean_text'] + q_text 
            #          + '\n'.join([exa['clean_text'][st:end] 
            #                       for st, end in answer_spans])
            # )
        
        
        features.append('')

        

def build_genqa_features(examples):
    contract_examples = group_by(examples, 'contract_index')
    questions = [(i,'This is the question of clause type i') for i in range(41)]
    all_features = []
    for c_examples in contract_examples:
        for cla_i, q_text in questions:
            c_features = build_genqa_features_for_contract(c_examples, cla_i, q_text)
            # can do sample here
            
            all_features.extend(c_features)


if __name__ == '__main__':
    import argparse
    import json
    import pickle
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--data_path')
    parser.add_argument('--save_path')
    parser.add_argument('--tokenizer')
    parser.add_argument('--n_cpu', type = int, default = 1)

    args = parser.parse_args()

    if args.task == 'get_examples':
        dataset = json.load(open(args.data_path))['data']
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents = True, exist_ok = True)
        examples = get_genqa_examples(dataset)
        with open(save_path, 'wb') as f:
            pickle.dump(examples, f)
    
    elif args.task == 'tokenize':
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents = True, exist_ok = True)

        examples = pickle.load(open(args.data_path, 'rb'))
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, use_fast = True, trust_remote_code = True
        )
        results = tokenize_genqa_examples(examples, tokenizer, num_cpu = args.n_cpu)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)