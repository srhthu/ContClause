"""
For each tokenizer, split the long paragraphs.

Keep the same hierarchical data format
"""

import json
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List, Tuple

from cont_gen.data_process.utils import overlap_of_two_span

def relocate_answer(answer, start, end):
    """
    Relocate the answer into span of [start, end]. Return None if no overlap.

    Args:
        answer: a dict of text, start_pos, end_pos
    """
    ol_pos = overlap_of_two_span((start, end), (answer['start_pos'], answer['end_pos']))
    # For most cases, ol_pos is equal to answer's (start_pos, end_pos)

    if ol_pos is None:
        return None
    
    new_text = answer['text'][ol_pos[0] - answer['start_pos']: ol_pos[1] + 1 - answer['start_pos']]
    # For most cases, this is new_text is equal to answer['text']

    return {
        'text': new_text,
        'start_pos': ol_pos[0] - start,
        'end_pos': ol_pos[1] - start,
    }

def relocate_qa(qa, start, end):
    """
    Find the answer span located in the provided spans and offset the position.

    Args:
        qa: a dict of
            q_id
            answers: a list of dict:
                text
                start_pos
                end_pos
        start: start index of new span
        end: end index of new span (included)
    """
    new_answers = [relocate_answer(answer, start, end) for answer in qa['answers']]
    new_answers = [k for k in new_answers if k is not None]
    if len(new_answers) > 0:
        return {'q_id': qa['q_id'], 'answers': new_answers}
    else:
        return None

def relocate_qas(qas, start, end):
    new_qas = [relocate_qa(qa, start, end) for qa in qas]
    new_qas = [k for k in new_qas if k is not None]
    return new_qas

def split_to_chunk_span(text, tokenizer: PreTrainedTokenizer, max_len):
    """
    Split text into chunks so that the token length of each chunk do not exceed max_len.

    Note: the chunks can be not adjacent.

    Return:
        span: a tuple of span range. (end NOT included)
    """
    enc = tokenizer(text, add_special_tokens=False)
    num_tokens = len(enc.input_ids)
    
    spans = []
    span_i = 0
    while span_i * max_len < num_tokens:
        sp_st = enc.token_to_chars(span_i * max_len).start
        span_last_token = min((span_i + 1) * max_len, num_tokens) - 1
        sp_ed = enc.token_to_chars(span_last_token).end

        spans.append((sp_st, sp_ed))

        span_i += 1

    return spans

def split_cuad_para(para_data, tokenizer, max_len):
    """
    Args:
        para_data: a dict of text, offset, qas
    """
    ctx = para_data['text']
    spans = split_to_chunk_span(ctx, tokenizer, max_len)

    new_paras = []
    for start, end in spans:
        qas = relocate_qas(para_data['qas'], start, end - 1)
        new_paras.append({
            'text': ctx[start: end],
            'offset': para_data['offset'] + start,
            'qas': qas
        })
    return new_paras

def process_doc_split_paras(doc_data, tokenizer, max_len):
    all_new_paras = []
    for pi, para_data in enumerate(doc_data['paras']):
        new_paras = split_cuad_para(para_data, tokenizer, max_len)
        for np in new_paras:
            np['old_para_idx'] = pi
        all_new_paras.extend(new_paras)
    return {'title': doc_data['title'], 'paras': all_new_paras}

if __name__ == '__main__':
    from pathlib import Path
    import json
    import argparse
    from tqdm import tqdm

    from cont_gen.utils import load_jsonl, save_jsonl

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help = 'file of para data')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer_name')
    parser.add_argument('max_len', type = int)
    args = parser.parse_args()
    
    all_docs = load_jsonl(args.input_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code = True)

    save_data = [
        process_doc_split_paras(d, tokenizer, args.max_len) for d in tqdm(all_docs)]
    
    
    save_jsonl(save_data)