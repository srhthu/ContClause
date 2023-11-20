"""
Build CUAD features for QA.
"""
import json
import pandas as pd

from collections import OrderedDict
import re
import numpy as np
from dotenv.main import dotenv_values
import os
import sys
import psutil
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer, RobertaTokenizerFast, RobertaModel
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from multiprocessing import Pool
import time
from copy import deepcopy
from functools import partial

from .utils import convert_token_char_map
from .basic import DocTokens

def create_examples(input_data):
    """
    Flat the nested examples.
    """
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"] # doc_id
        for pi, paragraph in enumerate(entry["paragraphs"]):
            # context_text = paragraph["context"]
            for cla_id, qa in enumerate(paragraph["qas"]):
                answer_spans = [[k['answer_start'],
                                 k['answer_start'] + len(k['text'])
                                 ] for k in qa['answers']]
                example = {
                    'doc_id': title,
                    'paragraph_id': pi,
                    'clause_id': cla_id,
                    'qa_id': qa['id'],
                    'question_text': qa["question"],
                    'is_impossible': qa.get("is_impossible", False),
                    'answer_spans': answer_spans, # end not included
                    'answer_texts': [k['text'] for k in qa['answers']]
                }

                examples.append(example)
    return examples

def convert_features(
    example, doc_obj,
    tokenizer: PreTrainedTokenizer,
    max_seq_length, 
    doc_stride, 
    max_query_length 
):
    """
    Convert a CUAD example to features
    """
    # Build features
    # Pad right: _bos <question> _sep <context>  _eos <pad>
    # Pad left： <pad> _eos <context> _sep <question> _bos

    sequence_added_tokens = len(tokenizer.build_inputs_with_special_tokens([]))
    sequence_pair_added_tokens = (
        len(tokenizer.build_inputs_with_special_tokens([99],[100])) - 2
    ) # handle issue with bert
    sep_token_num = sequence_pair_added_tokens - sequence_added_tokens

    # identify example answer span's position
    if not example['is_impossible']:
        # take the first span
        char_st, char_end = example['answer_spans'][0]
        # char_to_token = convert_token_char_map(doc_obj['token_to_char'], doc_len)
        char_to_token = doc_obj['char_to_token']
        answer_st = char_to_token[char_st][0]
        answer_end = char_to_token[char_end - 1][1] - 1 # char_end not included

        assert answer_st is not None and answer_end is not None


    # get spans
    spans = []
    query_token_ids = tokenizer.encode(
        example['question_text'], add_special_tokens=False, truncation=True, max_length=max_query_length
    )
    context_max_len = max_seq_length - len(query_token_ids) - sequence_pair_added_tokens
    assert context_max_len >= 100

    doc_tokens = doc_obj['doc_token_ids']
    
    for span_start in range(0, len(doc_tokens), doc_stride):

        context_token_ids = doc_tokens[span_start: span_start + context_max_len]
        
        num_pad = context_max_len - len(context_token_ids) # basically 0 except the last one

        encoded_dict = {}
        # pad to right
        input_ids = tokenizer.build_inputs_with_special_tokens(
            query_token_ids, context_token_ids
        )
        input_ids += [tokenizer.pad_token_id] * num_pad
        assert len(input_ids) == max_seq_length

        attention_mask = [1] * (max_seq_length - num_pad) + [0] * num_pad
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(
                query_token_ids, context_token_ids
        )
        token_type_ids += [0] * num_pad
        
        seq_len = max_seq_length - num_pad # length of non-pad input
        paragraph_len = len(context_token_ids) # length of the context

        # handle tokenizer without bos token, e.g., llama2
        has_bos = (sequence_added_tokens > 0)
        context_offset = int(has_bos) + len(query_token_ids) + sep_token_num # index of first context token
        
        
        # identify p_mask: 1 for token than cannot be in the answer
        p_mask = np.ones(max_seq_length)
        p_mask[context_offset: context_offset + paragraph_len] = 0
        cls_index = 0 # For determin negative spans. (padding right)

        # identify span start_position and end_position
        span_end = span_start + paragraph_len - 1 # included
        
        start_position = 0  # label for predict span start
        end_position = 0 # label for predict span end
        span_is_impossible = example['is_impossible']
        if not example['is_impossible']:
            out_of_span = not (answer_st >= span_start and answer_end <= span_end)
            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                start_position = answer_st - span_start + context_offset
                end_position = answer_end - span_start + context_offset

        encoded_dict = {}
        encoded_dict['input_ids'] = input_ids
        encoded_dict['attention_mask'] = attention_mask
        encoded_dict['token_type_ids'] = token_type_ids
        encoded_dict['seq_len'] = seq_len
        encoded_dict['paragraph_len'] = paragraph_len
        encoded_dict['context_offset'] = context_offset
        encoded_dict['span_start'] = span_start
        encoded_dict['p_mask'] = p_mask.tolist()
        encoded_dict['cls_index'] = cls_index
        encoded_dict['start_position'] = start_position
        encoded_dict['end_position'] = end_position
        encoded_dict['is_impossible'] = span_is_impossible
        # identifiers
        encoded_dict['doc_info'] = (example['doc_id'], example['clause_id'])
        encoded_dict['qa_id'] = example['qa_id']
        encoded_dict['example_index'] = 0 # be set later

        spans.append(encoded_dict)

        if span_end >= len(doc_tokens) - 1:
            # consume all tokens
            break
    
    return spans

def rand_choice(l, n):
    """choice number can be less than num of elements"""
    indexes = np.arange(len(l))
    np.random.shuffle(indexes)
    return [l[i] for i in indexes[:n]]

def sample_features(example, all_features):
    if example['is_impossible']:
        # add one negtive sample
        return rand_choice(all_features, 1)
    else:
        pos_f = [k for k in all_features if not k['is_impossible']]
        neg_f = [k for k in all_features if k['is_impossible']]
        if len(pos_f) == 0:
            # span is not extracted successfully
            return []
        if len(neg_f) > 0:
            neg_f_sel = rand_choice(neg_f, min(len(pos_f), len(neg_f)))
        else:
            neg_f_sel = []
        return pos_f + neg_f_sel

def process_features(
    examples, 
    doc_objs: List[Union[dict, DocTokens]], 
    tokenizer: PreTrainedTokenizer,
    max_seq_length, 
    doc_stride, 
    max_query_length 
):
    """
    convert all examples and do balance sampling.

    Args:
        kws: arguments for function convert_features
            max_seq_length
            doc_stride
            max_query_length 
    """
    doc_map = {k['doc_id']:k for k in doc_objs}

    all_features = []
    span_failed = []
    for i, e in enumerate(tqdm(examples, desc = 'Extract features')):
        feats = convert_features(
            e, doc_map[e['doc_id']],
            tokenizer = tokenizer,
            max_seq_length = max_seq_length,
            doc_stride = doc_stride,
            max_query_length = max_query_length
        )
        feats = sample_features(e, feats)

        for feat in feats:
            feat['example_index'] = i

        if not e['is_impossible'] and len(feats) == 0:
            span_failed.append(i)
        all_features.extend(feats)
    
    print(f'span failed to extract: {span_failed}')
    return all_features

def main():
    import argparse
    from pathlib import Path
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file')
    parser.add_argument('--doc_tk_path')
    parser.add_argument('--tokenizer_path')
    parser.add_argument('--output_path')
    parser.add_argument('--max_seq_length', type = int, default = 512)
    parser.add_argument('--doc_stride', type = int, default = 256)
    parser.add_argument('--max_query_length', type = int, default = 128)

    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(exist_ok=True)
    
    # load qa data
    qa_data = json.load(open(args.data_file))['data']

    print('Build examples')
    examples = create_examples(qa_data)

    # load doc str
    print('Load doc data')
    id2doc = pickle.load(open('./data/doc/doc_id_text.pkl', 'rb'))
    id2doc = {k['doc_id']: k['doc'] for k in id2doc}
    
    doc_objs = pickle.load(open(args.doc_tk_path, 'rb'))
    doc_objs = [DocTokens(**k, doc_text = id2doc[k['doc_id']]) for k in doc_objs]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, use_fast = True, trust_remote_code = True
    )

    features = process_features(
        examples, doc_objs, tokenizer = tokenizer,
        max_seq_length = args.max_seq_length,
        doc_stride = args.doc_stride,
        max_query_length = args.max_query_length
    )
    
    with open(args.output_path, 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':
    main()