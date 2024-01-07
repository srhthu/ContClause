"""
Split paragraphs into chunks to fit in a max token length
"""
import json
from transformers import AutoTokenizer
from typing import List, Tuple

from cont_gen.data_process.utils import overlap_of_two_span

def chunk_text(text, tokenizer, max_len):
    """
    Cut text into chunks of max token length of max_len.

    Return the chunk start and end positions
    """
    enc = tokenizer(text)

    chunk_pos: List[Tuple[int, int]] = [] # end pos is included
    span_char_start = 0
    span_token_start = enc.char_to_token(span_char_start)
    i = 0
    # traverse each character
    while i < len(text):
        if enc.char_to_token(i) - span_token_start >= max_len:
            # this char just overflow the max_len
            # add previous chars to span
            chunk_pos.append((span_char_start, i - 1))
            # start a new span
            span_char_start = i
            span_token_start = enc.char_to_token(i)
        # go to next char
        i += 1
    
    # add the final span
    chunk_pos.append((span_char_start, len(text) - 1))

    return chunk_pos

def split_one_para(para_data, tokenizer: AutoTokenizer, max_len):
    """Return a list of chunks with para_offset and qas"""
    text = para_data['text']

    # cut text into chunks
    chunk_pos = chunk_text(text, tokenizer, max_len)

    # get the qas for each chunk
    all_chunk_qas = []
    for start, end in chunk_pos:
        chunk_qas = []
        # traverse each qa to see whether it is in current chunk
        for qa in para_data['qas']:
            cur_answers = []
            for answer in qa['answers']:
                a_start, a_end = answer['start_pos'], answer['end_pos']
                ol_pos = overlap_of_two_span((start, end), (a_start, a_end))
                if ol_pos is None:
                    continue
                else:
                    # sub offset of chunk
                    cur_answers.append(
                        (ol_pos[0] - start, ol_pos[1] - start)
                    )
            if len(cur_answers)> 0:
                chunk_qas.append({
                    'qa_id': qa['qa_id'],
                    'q_id': qa['q_id'],
                    'answers': cur_answers
                })
        all_chunk_qas.append(chunk_qas)
    
    # yield chunk data
    # text, para_idx, para_offset, qas
    for ck_pos, ck_qas in zip(chunk_pos, all_chunk_qas):
        ck_text = text[ck_pos[0]: ck_pos[1] + 1]
        qas = [
            {
                'qa_id': qa['qa_id'],
                'q_id': qa['q_id'],
                'answers': [
                    {
                        'text': ck_text[a[0]: a[1] + 1],
                        'start_pos': a[0],
                        'end_pos': a[1]
                    } for a in qa['answers']
                ]
            } for qa in ck_qas
        ]
        
        ck_data = {
            'text': ck_text,
            'para_offset': ck_pos[0],
            'qas': qas
        }
        yield ck_data

def split_all_para_of_one_doc(doc_data, tokenizer, max_len):
    """
    split all paragraphs into chunks of one document. Add some id fields
    """
    title = doc_data['title']
    chunk_i = 0
    all_chunk = []
    for pi, para_data in enumerate(doc_data['paras']):
        for chunk in split_one_para(para_data, tokenizer, max_len):
            comp_chunk = {
                'title': title,
                'para_idx': pi,
                'para_offset': chunk['para_offset'],
                'chunk_index': chunk_i,
                'text': chunk['text'],
                'qas': chunk['qas']
            }
            all_chunk.append(comp_chunk)
            chunk_i += 1
    return all_chunk


if __name__ == '__main__':
    from pathlib import Path
    import json
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help = 'file of para data')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer_name')
    parser.add_argument('max_len', type = int)
    args = parser.parse_args()

    with open(args.input_path) as f:
        all_para_data = [json.loads(k) for k in f]
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code = True)

    chunks = []
    for doc_data in all_para_data:
        chunks.extend(split_all_para_of_one_doc(doc_data, tokenizer, args.max_len))
    
    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents = True, exist_ok = True)

    with open(out_p, 'w') as f:
        for d in chunks:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    