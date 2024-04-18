"""
Pre-process of documents and tokenization.

Pipeline:
    1. Keep at most 4 continuous spaces and 2 continuous newline
        Note: some tokenizers ignore space and newline while others not
    2. Tokenization given a tokenizer

Consider three relation between character and token:
    1-n: one character to multiple token, e.g., Chinese words, mu, alphabet with head
    n-1: several character compose one token
    1-0: some character has no corresponding tokens, e.g, space in BERT tk
Handle:
    from character to token: the end of cha should be the begin of next cha

"""
from pathlib import Path
import re
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from typing import List, Tuple, Dict, Optional, Union

from transformers import PreTrainedTokenizerFast, AutoTokenizer


def remove_space_keep_mapping(doc):
    """
    Return
        doc_clean
        clean_to_ori_span: List[Tuple[int, int]]
    """
    spans = [] # list of tuple: (start, end)
    patterns = [r'[ ]{2,}', r'[\n]{2,}']
    for pat in patterns:
        for m in re.finditer(pat, doc):
            spans.append(m.span())
    spans = sorted(spans, key = lambda k: k[0])
    # check no overlap
    for i in range(1, len(spans)):
        assert spans[i-1][1] <= spans[i][0]
    
    # process each span
    prev_end = 0
    clean_text = []
    
    clean_to_ori_span = [] # map from clean index to original span of (start, end)

    for start, end in spans:
        # add characters before span
        s_pre = doc[prev_end:start]
        clean_text.append(s_pre)
        for i in range(prev_end, start):
            clean_to_ori_span.append([i, i+1])
        # compress span to one character
        # use span's first chr to represent the span
        clean_text.append(doc[start])
        clean_to_ori_span.append([start, end])

        prev_end = end

    if prev_end < len(doc):
        # add the last normal span
        clean_text.append(doc[prev_end:])
        for i in range(prev_end, len(doc)):
            clean_to_ori_span.append([i, i+1])
    return ''.join(clean_text), clean_to_ori_span
        

def doc_tokenization_fast(doc, tokenizer: PreTrainedTokenizerFast, remove_space: bool):
    """
    Use FastTokenize to tokenize the document and get the char to token mapping

    Return:
        doc_token_ids: List[int]
        token_to_char (List[Tuple[int, int]]): token's span position
            the end position is not included
    """
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    if remove_space:
        doc_clean, clean_to_ori_span = remove_space_keep_mapping(doc)
    else:
        doc_clean = doc

    enc = tokenizer(
        doc_clean, truncation = False, add_special_tokens = False, verbose = False
    )
    doc_token_ids = enc.input_ids
    token_to_char: List[Tuple[int, int]] = [
        list(enc.token_to_chars(i)) 
            for i in range(len(doc_token_ids))
    ] # the end not included
    if remove_space:
        for i in range(len(doc_token_ids)):
            clean_st, clean_end = token_to_char[i]
            ori_st = clean_to_ori_span[clean_st][0]
            ori_end = clean_to_ori_span[clean_end - 1][1]
            token_to_char[i] = [ori_st, ori_end]
    
    return doc_token_ids, token_to_char


def convert_all_documents(docs, tokenizer, remove_space:bool, num_cpu = 1):
    """Tokenize documents with multi processes"""
    results = []
    _func = partial(
        doc_tokenization_fast, 
        tokenizer = tokenizer, 
        remove_space = remove_space
    )
    with Pool(num_cpu) as p:
        bar = tqdm(total = len(docs),desc = 'Doc Tokenization')
        for (doc_token_ids, token_to_char) in p.imap(_func, docs):
            results.append({
                'doc_token_ids': doc_token_ids, 
                'token_to_char': token_to_char, 
            })
            bar.update()
        bar.close()
    
    return results

def process_cuad_data(qa_data, tokenizer, remove_space, num_cpu):
    """
    Assign each document its title as id and conduct tokenization.

    Return:
        doc_objs: each doc_obj is a dict of
            doc_id: str
            doc_token_ids: List[int]
            token_to_char: List[Tuple[int, int]]
    """
    doc_ids = [e['title'] for e in qa_data]
    docs = [e['paragraphs'][0]['context'] for e in qa_data]

    doc_objs = convert_all_documents(docs, tokenizer, remove_space, num_cpu)
    doc_objs = [{'doc_id':did, **dobj} for did, dobj in zip(doc_ids, doc_objs)]

    return doc_objs

def main():
    import argparse
    import json
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('cuad_file')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer_path')
    parser.add_argument('--remove_space', action='store_true')
    parser.add_argument('--n_cpu', type = int, default = 1)

    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(exist_ok=True)

    qa_data = json.load(open(args.cuad_file))['data']
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, use_fast = True, trust_remote_code = True
    )
    doc_objs = process_cuad_data(
        qa_data, tokenizer, 
        remove_space = args.remove_space,
        num_cpu = args.n_cpu
    )

    with open(args.output_path, 'wb') as f:
        pickle.dump(doc_objs, f)

if __name__ == '__main__':
    main()