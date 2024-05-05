"""
Split the whole document into chunks.

1. identify paragraphs with their start positions. each paragraph is seemd as a candidate chunk.
2. find long paragraphs, and split it into chunks
3. Merge chunks.
4. For each chunk, determin the clauses.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

from transformers import PreTrainedTokenizer

from cont_gen.data_process.utils import (
    cut_spans_return_offset
)

def split_doc_to_paragraphs(doc: str):
    """
    Split document by newline.

    Return:
        paras: a list of dict:
            text
            offset
    """
    pat = r'[\n]+'
    break_spans = [m.span() for m in re.finditer(pat, doc)]
    text_parts, start_indexes = cut_spans_return_offset(doc, break_spans)

    return [{'text': t, 'offset': o} for t, o in zip(text_parts, start_indexes)]


def split_one_para_to_chunks(text, tokenizer: PreTrainedTokenizer, max_len):
    """
    Split text into chunks so that the token length of each chunk do not exceed max_len.

    Note: the chunks can be not adjacent.

    Return:
        chunks: a list of dict:
            text: (str)
            offset: (int)
            num_tokens: (int)
    """
    enc = tokenizer(text, add_special_tokens=False)
    num_tokens = len(enc.input_ids)
    if num_tokens <= max_len:
        return [{'text': text, 'offset': 0, 'num_tokens': num_tokens}]
    
    chunks = []
    span_i = 0
    while (span_i + 1) * max_len <= num_tokens:
        sp_st = enc.token_to_chars(span_i * max_len).start
        span_last_token = min((span_i + 1 * max_len), num_tokens) - 1
        sp_ed = enc.token_to_chars(span_last_token).end

        chunk = {
                'text': text[sp_st: sp_ed],
                'offset': sp_st,
                'num_tokens': span_last_token - span_i * max_len + 1
            }
        chunks.append(chunk)

    return chunks

def merge_chunks(chunks, max_len):
    """
    Args:
        chunks: a list of dict with keys of text, offset, num_tokens
        max_len: max number of tokens in one chunk.
    
    Return:
        chunk_spans
    """
    # group chunks
    groups = []
    cur_group = []
    cur_len = 0
    for chunk in chunks:
        if chunk['num_tokens'] + cur_len <= max_len:
            cur_group.append(chunk)
            cur_len = cur_len + chunk['num_tokens']
        else:
            if len(cur_group) == 0:
                groups.append([chunk])
            else:
                groups.append(cur_group)
                cur_group = [chunk]
                cur_len = chunk['num_tokens']
    groups.append(cur_group)

    chunk_spans = []
    for group in groups:
        start = group[0]['offset']
        end = group[-1]['offset'] + len(group[-1]['text'])
        chunk_spans.append((start, end))
    
    return chunk_spans