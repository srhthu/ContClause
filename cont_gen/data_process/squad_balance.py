"""
Process a squad dataset with relatively balenced positive and negative samples.
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

@dataclass
class SquadExample:
    qas_id: str  # in the form of {title}_{clause}
    question_text: str  # question
    context_text: str # contract paragraph
    answer_text: str  # answer span text
    start_position_character: str # the original start
    title: str
    is_impossible: bool
    answers: List  # Empty for train. For eval, it is List of dict with `text` and `answer_start`
    paragraph_id: int # 0 in the CUAD dataset as the only paragraph is the whole contract

    def get_span(self):
        """Span position of characters. End position is included"""
        if self.is_impossible:
            return [None, None]
        return [self.start_position_character, self.start_position_character + len(self.answer_text) - 1]

@dataclass
class Document:
    """A document holds tokenized tokens and the position mapping"""
    doc: int
    doc_tokens: List[str]
    token_to_char: List[Tuple[int, int]]
    char_to_token: List[int]

def create_examples(input_data, set_type)-> List[SquadExample]:
    """Extract QA examples from original data"""
    is_training = set_type == "train"
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for pi, paragraph in enumerate(entry["paragraphs"]):
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                is_impossible = qa.get("is_impossible", False)
                if not is_impossible:
                    if is_training:
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                    else:
                        answers = qa["answers"]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                    paragraph_id = pi
                )
                examples.append(example)
    return examples

# Utilities related to document
def get_all_doc(qa_data) -> List[Tuple[str, str]]:
    doc_id_content_list = []
    for entry in qa_data:
        title = entry['title']
        for i, paragraph in enumerate(entry['paragraphs']):
            doc_id_content_list.append([f'{title}_{i}', paragraph['context']])
    return doc_id_content_list

def doc_tokenization(doc, tokenizer: PreTrainedTokenizerFast):
    """Tokenize the document and get the char to token mapping"""
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    enc = tokenizer(doc, truncation = False, add_special_tokens = False, verbose = False)
    doc_tokens = tokenizer.convert_ids_to_tokens(enc.input_ids)
    token_to_char: List[Tuple[int, int]] = [enc.token_to_chars(i) for i in range(len(doc_tokens))] # the end not included
    char_to_token: List[int] = [enc.char_to_token(i) for i in range(len(doc))]
    return doc_tokens, token_to_char, char_to_token

def convert_all_documents(doc_id_contents, tokenizer, num_cpu = 1):
    doc_ids = [k[0] for k in doc_id_contents]
    docs = [k[1] for k in doc_id_contents]
    pro_docs = []
    with Pool(num_cpu) as p:
        bar = tqdm(
            total = len(docs),
            desc = 'Doc Tokenization'
        )
        for i, (doc_tokens, token_to_char, char_to_token) in enumerate(
            p.imap(partial(doc_tokenization, tokenizer = tokenizer), docs)
        ):
            pro_docs.append(Document(docs[i], doc_tokens, token_to_char, char_to_token))
            bar.update()
        bar.close()
    
    doc_map = {did: d for did, d in zip(doc_ids, pro_docs)}
    return doc_map

def check_answer_exist(example):
    if example.is_impossible:
        return True
    st_pos_chr = example.start_position_character
    end_pos_chr = st_pos_chr + len(example.answer_text)
    span_text = example.context_text[st_pos_chr: end_pos_chr]
    return span_text == example.answer_text
    

def convert_features(example, max_seq_length, doc_obj, doc_stride, max_query_length, is_training, tokenizer: PreTrainedTokenizer):
    # Build features
    # _bos <question> _sep <context>  _eos <pad>
    # <pad> _eos <context> _sep <question> _bos

    sequence_added_tokens = len(tokenizer.build_inputs_with_special_tokens([]))
    sequence_pair_added_tokens = len(tokenizer.build_inputs_with_special_tokens([],[]))
    sep_token_num = sequence_pair_added_tokens - sequence_added_tokens

    # identify example answer span's position
    if is_training and not example.is_impossible:
        st_pos_chr, end_pos_chr = example.get_span()
        tok_start_position = doc_obj.char_to_token[st_pos_chr]
        tok_end_position = doc_obj.char_to_token[end_pos_chr]
        assert tok_start_position is not None and tok_end_position is not None


    # get spans
    spans = []
    query_token_ids = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )
    context_max_len = max_seq_length - len(query_token_ids) - sequence_pair_added_tokens
    assert context_max_len >= 100

    doc_tokens = doc_obj.doc_tokens
    bos_ = [tokenizer.bos_token_id]
    eos_ = [tokenizer.eos_token_id]
    sep_ = [tokenizer.sep_token_id] * 2
    
    for span_start in range(0, len(doc_tokens), doc_stride):

        context_token_ids = tokenizer.convert_tokens_to_ids(
            doc_tokens[span_start: span_start + context_max_len]
        )
        num_pad = context_max_len - len(context_token_ids)

        encoded_dict = {}
        # pad to right
        input_ids = (
            bos_ + query_token_ids + sep_ 
            + context_token_ids + eos_
            + [tokenizer.pad_token_id] * num_pad
        )
        assert len(input_ids) == max_seq_length
        attention_mask = [1] * (max_seq_length - num_pad) + [0] * num_pad
        token_type_ids = (
            tokenizer.create_token_type_ids_from_sequences(
                query_token_ids, context_token_ids
            ) 
            + [0] * num_pad
        )
        # non_padded_ids = input_ids[: max_seq_length - num_pad]
        # tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
        # use the seq_len as the proxy of tokens
        seq_len = max_seq_length - num_pad
        
        paragraph_len = len(context_token_ids) # length of the context
        truncated_query_with_special_tokens_length = (
            1 + len(query_token_ids) + sep_token_num
         ) # indicate the  number of tokens ahead of context

        token_to_orig_map = {} # the token position in the original doc tokens
        for i in range(paragraph_len):
            index = truncated_query_with_special_tokens_length + i
            token_to_orig_map[index] = span_start + i
            # [TODO] check how this will be used in future
        
        # identify p_mask: 1 for token than cannot be in the answer
        p_mask = np.ones(max_seq_length)
        p_mask[
            truncated_query_with_special_tokens_length: 
            truncated_query_with_special_tokens_length + paragraph_len
        ] = 0
        cls_index = 0 # padding right

        # identify span start_position and end_position
        span_end = span_start + paragraph_len - 1 # included
        
        start_position = 0
        end_position = 0
        span_is_impossible = example.is_impossible
        if is_training and not example.is_impossible:
            out_of_span = not (
                tok_start_position >= span_start 
                and tok_end_position <= span_end
            )
            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                doc_offset = truncated_query_with_special_tokens_length # for padding right
                start_position = tok_start_position - span_start + doc_offset
                end_position = tok_end_position - span_start + doc_offset

        encoded_dict = {}
        encoded_dict['input_ids'] = input_ids
        encoded_dict['attention_mask'] = attention_mask
        encoded_dict['token_type_ids'] = token_type_ids
        encoded_dict['seq_len'] = seq_len
        encoded_dict['paragraph_len'] = paragraph_len
        encoded_dict['truncated_query_with_special_tokens_length'] = truncated_query_with_special_tokens_length
        encoded_dict['token_to_orig_map'] = token_to_orig_map
        encoded_dict['start'] = span_start
        # span_end = span_start + paragraph_len
        encoded_dict['p_mask'] = p_mask.tolist()
        encoded_dict['cls_index'] = cls_index
        encoded_dict['start_position'] = start_position
        encoded_dict['end_position'] = end_position
        encoded_dict['is_impossible'] = span_is_impossible
        encoded_dict['qas_id'] = example.qas_id

        spans.append(encoded_dict)

        if span_end >= len(doc_obj.doc_tokens) - 1:
            # consume all tokens
            break
    
    return spans

def rand_choice(l, n):
    indexes = np.arange(len(l))
    np.random.shuffle(indexes)
    return [l[i] for i in indexes[:n]]


class CUAD_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, set_type, doc_stride, max_query_length, max_seq_length, num_cpu):
        self.tokenizer = tokenizer
        self.set_type = set_type
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.num_cpu = num_cpu
        # debug
        self._start_is_not_word = []
        self._end_is_not_word = []
        self._has_span_chunk_num = []
        self._missing_answer = []

        # load data
        qa_data = json.load(open(path))['data'][:20]
        print(f'Num of data: {len(qa_data)}')
        self.qa_data = qa_data
        
        # create examples
        self.examples = create_examples(qa_data, set_type)

        self.process()
    
    def process(self):
        self.process_docs()
        self.build_features()
    
    def process_docs(self):
        self.doc_id_contents = get_all_doc(self.qa_data)
        self.doc_map = convert_all_documents(
            self.doc_id_contents, self.tokenizer, num_cpu = self.num_cpu
        )
    
    def convert_feaures_with_sampling(self, example):
        feas = convert_features(
            example,
            max_seq_length = self.max_seq_length,
            doc_obj = self.doc_map[f'{example.title}_{example.paragraph_id}'], 
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=True,
            tokenizer = self.tokenizer
        )
        if example.is_impossible:
            # add one negtive sample
            return rand_choice(feas, 1)
        else:
            pos_f = [k for k in feas if not k['is_impossible']]
            neg_f = [k for k in feas if k['is_impossible']]
            if len(pos_f) == 0:
                # span is not extracted successfully
                return []
            if len(neg_f) > 0:
                neg_f_sel = rand_choice(neg_f, min(len(pos_f), len(neg_f)))
            else:
                neg_f_sel = []
            return pos_f + neg_f_sel

    def build_features(self):
        # create features
        features = []
        span_failed = []
        for i, e in enumerate(tqdm(self.examples, desc = 'Extract features')):
            results = self.convert_feaures_with_sampling(e)
            features.extend(results)

            if not e.is_impossible and len(results) == 0:
                span_failed.append(i)

        self.features = features
        # debug
        print(f'span failed to extract: {span_failed}')
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

