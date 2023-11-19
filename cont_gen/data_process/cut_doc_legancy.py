"""
Pre-process of documents and tokenization.

There are some potential issues.
    Cannot exactly map subwords to original spans.
        subwords can have extra characters, e.g., #.
"""
import json
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import re
import numpy as np
from dotenv.main import dotenv_values
import os
import sys
import psutil
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from multiprocessing import Pool
import time
from copy import deepcopy
from functools import partial
import torch
import random

def get_all_doc(qa_data) -> List[Tuple[str, str]]:
    """Return the list of tuple: ({title}_{p_id}, context)"""
    doc_id_content_list = []
    for entry in qa_data:
        title = entry['title']
        for i, paragraph in enumerate(entry['paragraphs']):
            doc_id_content_list.append([f'{title}_{i}', paragraph['context']])
    return doc_id_content_list
        

def doc_tokenization_fast(doc, tokenizer: PreTrainedTokenizerFast):
    """
    Use FastTokenize to tokenize the document and get the char to token mapping

    Return:
        doc_token_ids: List[int]
        token_to_char (List[Tuple[int, int]]): token's span position
            the end position is not included
        char_to_token (List[int]): token index of a character, may be None
    """
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    enc = tokenizer(doc, truncation = False, add_special_tokens = False, verbose = False)
    doc_token_ids = enc.input_ids
    token_to_char: List[Tuple[int, int]] = [enc.token_to_chars(i) for i in range(len(doc_token_ids))] # the end not included
    char_to_token: List[int] = [enc.char_to_token(i) for i in range(len(doc))]
    return doc_token_ids, token_to_char, char_to_token

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def split_by_whitespace(doc):
    """
    Return words and char_to_word offset.
    whitespace offset is set to None.
    """
    doc_tokens = []
    char_to_word_offset = []
    word_to_char = []
    prev_is_whitespace = True
    for ci, c in enumerate(doc):
        if _is_whitespace(c):
            prev_is_whitespace = True
            char_to_word_offset.append(None)
            if len(word_to_char) > 0 and word_to_char[-1][1] is None:
                word_to_char[-1][1] = ci
        else:
            if prev_is_whitespace:
                # start of a new word
                doc_tokens.append(c)
                word_to_char.append([ci, None])
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
    # add the end position of the last word
    if len(word_to_char) > 0 and word_to_char[-1][1] is None:
        word_to_char[-1][1] = len(doc)
    
    return doc_tokens, char_to_word_offset, word_to_char

def word_to_subword(
    doc_words: List[str], 
    tokenizer: PreTrainedTokenizer,
    doc_str: str,
    word_to_char
):
    """
    Convert words to tokens and maintain the position mapping.
    """
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    token_to_char = []

    for i, token in enumerate(doc_words):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
        char_st, char_end = word_to_char[i]
        span = doc_str[char_st: char_end]
        # Note: tokenizer is initialized with add_prefix_space = True

        # fix the bug for uncased tokenizer
        if hasattr(tokenizer, 'do_lower_case') and tokenizer.do_lower_case:
            span = span.lower()

        match_offset = 0 # previously matched tokens. Incase two sub_tokens are same
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
            t = tokenizer.decode([sub_token])
            # handle some special cases
            # handle decode adding space
            if t.startswith(' ') and span[match_offset] != ' ':
                t = t[1:]
            # handle decode adding ##
            if span[match_offset] != '#' and t.startswith('#'):
                t = t.lstrip('##')

            try:
                t_st = span[match_offset:].index(t)
            except:
                print(span)
                print(span[match_offset:])
                print(t)
                raise RuntimeError

            token_to_char.append([
                char_st + match_offset +  t_st, 
                char_st + match_offset + t_st + len(t)]
            )
            match_offset += len(t)
        
    return all_doc_tokens, token_to_char

def convert_mapping_from_token(token_to_char, length):
    """
    Given the token_to_char map, return the char_to_token map.
    For space that not in token, return None.
    """
    char_to_token = [None for _ in range(length)]
    for i, (start, end) in enumerate(token_to_char):
        for j in range(start, end):
            assert char_to_token[j] is None
            char_to_token[j] = i
    return char_to_token

def doc_tokenization_by_word(doc, tokenizer: PreTrainedTokenizer):
    """
    Remove space and tokenize by word.

    Return same as doc_tokenization_fast
    """
    doc_words, char_to_word_offset, word_to_char = split_by_whitespace(doc)
    doc_tokens, token_to_char = word_to_subword(doc_words, tokenizer, doc, word_to_char)
    char_to_token = convert_mapping_from_token(token_to_char, len(doc))

    return doc_tokens, token_to_char, char_to_token

def convert_all_documents(doc_id_contents, tokenizer, remove_space:bool, num_cpu = 1):
    doc_ids = [k[0] for k in doc_id_contents]
    docs = [k[1] for k in doc_id_contents]
    tk_func = doc_tokenization_by_word if remove_space else doc_tokenization_fast

    results = []
    with Pool(num_cpu) as p:
        bar = tqdm(
            total = len(docs),
            desc = 'Doc Tokenization'
        )
        for i, (doc_tokens, token_to_char, char_to_token) in enumerate(
            p.imap(partial(tk_func, tokenizer = tokenizer), docs)
        ):
            results.append({
                'doc_id': doc_ids[i],
                'doc_tokens': doc_tokens, 
                'token_to_char': token_to_char, 
                'char_to_token': char_to_token
            })
            bar.update()
        bar.close()
    
    doc_map = {did: d for did, d in zip(doc_ids, results)}
    return doc_map