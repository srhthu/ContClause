"""
Process the squad qa format data.

What should we get?
- mapping from token to original character position

Process:
    1. Split passage into chuncks, by stride
        1.1 get the chunck token position mapping
    2. For each chunk, determin the answer relative position
"""
# %%
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
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer, RobertaTokenizerFast, RobertaModel
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from multiprocessing import Pool
import time
from copy import deepcopy
from functools import partial
import matplotlib.pyplot as plt
# %%
ENVS = dotenv_values('../.env')
cuad_dir = Path(ENVS['CUAD_PATH'])

# %%
def print_mem():
    pid = os.getpid()
    mem = psutil.Process(pid).memory_info().rss / 1024 / 1024 / 1024
    print(f'PID {pid}: {mem:.2f}GB')
# %%
qa_clauses = json.load(open(cuad_dir / 'CUAD_v1.json'))
qa_data = qa_clauses['data']
# %%
print_mem()
# %%
@dataclass
class SquadExample:
    qas_id: str
    question_text: str
    context_text: str
    answer_text: str
    start_position_character: str
    title: str
    is_impossible: bool
    answers: List
    paragraph_id: int

    def get_span(self):
        """End position is included"""
        if self.is_impossible:
            return [None, None]
        return [self.start_position_character, self.start_position_character + len(self.answer_text) - 1]
    
    def doc_id(self):
        return f'{self.title}_{self.paragraph_id}'

@dataclass
class Document:
    doc: str
    doc_tokens: List[str]
    token_to_char: List[Tuple[int, int]]
    char_to_token: List[int]
    # chunks: List[List[str]]
    # chunk_start_pos: List[int]

@dataclass
class CUAD_Document:
    doc: str  # original document
    doc_words: List[str] # whitespace tokenized
    doc_tokens: List[str] # done by tokenizer
    char_to_word_offset: List[int] # from doc to doc_words
    tok_to_orig_index: List[int] # from doc_tokens to doc_words
    orig_to_tok_index: List[int] # from doc_words to doc_tokens
    

def create_examples(input_data, set_type):
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


# def doc_tokenization(doc, tokenizer: PreTrainedTokenizerFast):
#     """Tokenize the document and get the char to token mapping"""
#     enc = tokenizer(doc, truncation = False, add_special_tokens = False, verbose = False)
#     doc_tokens = tokenizer.convert_ids_to_tokens(enc.input_ids)
#     token_to_char: List[Tuple[int, int]] = [enc.token_to_chars(i) for i in range(len(doc_tokens))] # the end not included
#     char_to_token: List[int] = [enc.char_to_token(i) for i in range(len(doc))]
#     return doc_tokens, token_to_char, char_to_token

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def split_by_whitespace(doc):
    """Return words and char_to_word offset"""
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in doc:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset

def word_to_subword(doc_words, tokenizer: PreTrainedTokenizer):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for i, token in enumerate(doc_words):
        orig_to_tok_index.append(len(all_doc_tokens))
        # if tokenizer.__class__.__name__ in [
        #     "RobertaTokenizer",
        #     "LongformerTokenizer",
        #     "BartTokenizer",
        #     "RobertaTokenizerFast",
        #     "LongformerTokenizerFast",
        #     "BartTokenizerFast",
        # ]:
        #     sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        # else:
        #     sub_tokens = tokenizer.tokenize(token)
        sub_tokens = tokenizer.tokenize(token) 
        # Note: tokenizer is initialized with add_prefix_space = True
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    return all_doc_tokens, tok_to_orig_index, orig_to_tok_index

def doc_to_obj(doc, tokenizer)->CUAD_Document:
    """Tokenize a document"""
    doc_words, char_to_word_offset = split_by_whitespace(doc)
    doc_tokens, tok_to_orig_index, orig_to_tok_index = word_to_subword(doc_words, tokenizer)
    return CUAD_Document(
        doc, doc_words, doc_tokens, 
        char_to_word_offset, tok_to_orig_index, orig_to_tok_index
    )

def get_all_doc(qa_data) -> List[Tuple[str, str]]:
    doc_id_content_list = []
    for entry in qa_data:
        title = entry['title']
        for i, paragraph in enumerate(entry['paragraphs']):
            doc_id_content_list.append([f'{title}_{i}', paragraph['context']])
    return doc_id_content_list

def process_all_documents(qa_data, tokenizer, num_cpu = 1):
    doc_id_contents = get_all_doc(qa_data)
    doc_ids = [k[0] for k in doc_id_contents]
    docs = [k[1] for k in doc_id_contents]
    doc_map = {}
    with Pool(num_cpu) as p:
        bar = tqdm(total = len(docs), desc = 'Doc Tokenization')
        for i, doc_obj in enumerate(
            p.imap(partial(doc_to_obj, tokenizer = tokenizer), docs)
        ):
            doc_map[doc_ids[i]] = doc_obj
            bar.update()
        bar.close()
    return doc_map


def has_span(span_start, span_end, chunk_start, chunk_end):
    """
    Determin whether a span range is in the chunk.
        [---chunk---]
    [---span---]
        [---chunk---]
            [---span---]
    """
    if span_end <= chunk_start:
        return False
    if span_end <= chunk_end:
        return True
    if span_start <= chunk_end - 1:
        return True
    return False

def check_answer_exist(example):
    if example.is_impossible:
        return True
    st_pos_chr = example.start_position_character
    end_pos_chr = st_pos_chr + len(example.answer_text)
    span_text = example.context_text[st_pos_chr: end_pos_chr]
    return span_text == example.answer_text
    

def convert_features(example, doc_obj: CUAD_Document, max_seq_length, doc_stride, max_query_length, is_training, tokenizer: PreTrainedTokenizer):
    # Build features
    # _bos <question> _sep <context>  _eos <pad>
    # <pad> _eos <context> _sep <question> _bos

    sequence_added_tokens = len(tokenizer.build_inputs_with_special_tokens([]))
    sequence_pair_added_tokens = len(tokenizer.build_inputs_with_special_tokens([],[]))
    sep_token_num = sequence_pair_added_tokens - sequence_added_tokens

    # identify example answer span's position
    if is_training and not example.is_impossible:
        st_pos_chr, end_pos_chr = example.get_span()
        word_st_pos = doc_obj.char_to_word_offset[st_pos_chr]
        word_end_pos = doc_obj.char_to_word_offset[end_pos_chr]
        tok_start_position = doc_obj.orig_to_tok_index[word_st_pos]
        tok_end_position = doc_obj.orig_to_tok_index[word_end_pos]
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
        context_offset = truncated_query_with_special_tokens_length # rename

        token_to_orig_map = {} # the token position in the original doc tokens
        for i in range(paragraph_len):
            index = truncated_query_with_special_tokens_length + i
            token_to_orig_map[index] = span_start + i
            # [TODO] check how this will be used in future
        
        # identify p_mask: 1 for token than cannot be in the answer
        p_mask = np.ones(max_seq_length)
        p_mask[context_offset: context_offset + paragraph_len] = 0
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
                start_position = tok_start_position - span_start + context_offset
                end_position = tok_end_position - span_start + context_offset

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
        encoded_dict['example_index'] = 0
        encoded_dict['unique_id'] = 0

        spans.append(encoded_dict)

        if span_end >= len(doc_obj.doc_tokens) - 1:
            # consume all tokens
            break
    
    return spans

def rand_choice(l, n):
    indexes = np.arange(len(l))
    np.random.shuffle(indexes)
    return [l[i] for i in indexes[:n]]

def convert_feaures_with_sampling(
    example, doc_obj, 
    max_seq_length, doc_stride, max_query_length, is_training, 
    tokenizer):
    feas = convert_features(
        example,
        doc_obj,
        max_seq_length = max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training,
        tokenizer = tokenizer
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

def process_features(
    examples, doc_map, 
    max_seq_length, doc_stride, max_query_length, is_training, 
    tokenizer: PreTrainedTokenizer):
    features = []
    span_failed = []
    for i, e in enumerate(tqdm(examples, desc = 'Extract features')):
        results = convert_feaures_with_sampling(
            e, doc_map[e.doc_id()],
            max_seq_length, doc_stride, max_query_length, is_training,
            tokenizer
        )
        features.extend(results)

        if not e.is_impossible and len(results) == 0:
            span_failed.append(i)

    print(f'span failed to extract: {span_failed}')
    return features

class CUAD_Dataset:
    def __init__(self, path, tokenizer, set_type, doc_stride, chunk_size):
        self.tokenizer = tokenizer
        self.set_type = set_type
        self.doc_stride = doc_stride
        self.chunk_size = chunk_size
        # debug
        self._start_is_not_word = []
        self._end_is_not_word = []
        self._has_span_chunk_num = []
        self._missing_answer = []

        # load data
        qa_data = json.load(open(path))['data'][:20]

        # Hold a map of paragraphs, so that each paragraph is processed once for many questions
        # map from <title>_<para id> to tokenized and chunked paragraph context
        doc_map: Dict[str, Document] = OrderedDict() 
        # create doc tokenization tasks
        doc_id_content_list = []
        for entry in qa_data:
            title = entry['title']
            for i, paragraph in enumerate(entry['paragraphs']):
                doc_id_content_list.append([f'{title}_{i}', paragraph['context']])
        
        # doc_id_content_list.sort(key = lambda k: -len(k[1]))
        self.doc_id_content_list = doc_id_content_list
                
        print(f'Num of doc: {len(doc_id_content_list)}')
        self.qa_data = qa_data
        self.doc_map = doc_map
        
        # create examples
        self.examples = create_examples(qa_data, set_type)

    
    def process_doc(self, doc):
        # tokenization and mapping
        doc_tokens, token_to_char, char_to_token = doc_tokenization(doc, self.tokenizer)

        # do chunk
        # chunks, chunk_start_pos = doc_partition(doc_tokens, self.doc_stride, self.chunk_size)
        return Document(
            doc = doc, doc_tokens=doc_tokens, token_to_char=token_to_char,
            char_to_token = char_to_token, 
            # chunks = chunks, chunk_start_pos = chunk_start_pos
        )

    def do_tokenization(self, num_cpu):
        """Use imap of multiprocess"""
        doc_ids = [k[0] for k in self.doc_id_content_list]
        docs = [k[1] for k in self.doc_id_content_list]
        with Pool(num_cpu) as p:
            pro_docs = list(
                tqdm(
                    p.imap(self.process_doc, docs),
                    total = len(docs)
                )
            )
        self.doc_map = {did: d for did, d in zip(doc_ids, pro_docs)}
    
    def do_tokenization_2(self, num_cpu):
        """Use map of multiprocess"""
        doc_ids = [k[0] for k in self.doc_id_content_list]
        docs = [k[1] for k in self.doc_id_content_list]
        start = time.time()
        with Pool(num_cpu) as p:
            pro_docs = p.map(self.process_doc, docs)
        end = time.time()
        print(f'Duration: {end - start:.2f}s')
        self.doc_map = {did: d for did, d in zip(doc_ids, pro_docs)}

    def do_build_features(self):
        # create features
        self.features = [fea for ex in self.examples for fea in self.convert_features[ex]]

        # debug
        print(f'missing answer: {len(self._missing_answer)}')
        print(f'answer chunk number: {np.mean(self._has_span_chunk_num)}')


# %%
tk = AutoTokenizer.from_pretrained('roberta-base', use_fast = True, verbose = False)
tk_nf = AutoTokenizer.from_pretrained('roberta-base', use_fast = False, verbose = False)
# %%
tk_pre = AutoTokenizer.from_pretrained('roberta-base', use_fast = True, add_prefix_space = True)
# %%
examples = create_examples(qa_data, 'train')
# %%
doc_map = process_all_documents(qa_data, tk_pre, 5)
# %%
# statistics of doc len
doc_lens = [len(doc_o.doc_tokens) for doc_o in doc_map.values()]
_ = plt.hist(doc_lens, bins = 20)
# %%
features = process_features(
    examples, doc_map, 
    max_seq_length=512, 
    doc_stride = 256,
    max_query_length=128,
    is_training=True,
    tokenizer = tk
)
# %%
ds = CUAD_Dataset(cuad_dir / 'CUAD_v1.json', tk, 'train', 128, 400)
# %%
ds.do_tokenization_2(15)

# %%
examples = create_examples(ds.qa_data, 'train')
# %%
feas = convert_features(
    examples[20], 
    max_seq_length=512,
    doc_map = ds.doc_map, 
    doc_stride=256,
    max_query_length=128,
    is_training=True,
    tokenizer = tk)
# %%
all_feas = []
no_pos_span = []
for i, e in enumerate(tqdm(examples, total = len(examples))):
    feas = convert_features(
        e,
        max_seq_length=512,
        doc_map = ds.doc_map, 
        doc_stride=256,
        max_query_length=128,
        is_training=True,
        tokenizer = tk
    )
    if e.is_impossible:
        # add one negtive sample
        all_feas.append(np.random.choice(feas))
    else:
        pos_f = [k for k in feas if not k['is_impossible']]
        neg_f = [k for k in feas if k['is_impossible']]
        if len(pos_f) == 0:
            no_pos_span.append(i)
            continue
        if len(neg_f) > 0:
            neg_f_sel = np.random.choice(neg_f, min(len(pos_f), len(neg_f)), replace = False)
        else:
            neg_f_sel = []
        all_feas.extend(pos_f)
        all_feas.extend(neg_f_sel)
    
    # all_feas.extend(feas)
    if i+1 >= 100:
        ...
    
# %%
from transformers.data.processors.squad import squad_convert_example_to_features, squad_convert_example_to_features_init, SquadProcessor

squad_convert_example_to_features_init(tk_nf)

# %%

# %%
text = "Hello world!"
tokens = tk_nf.tokenize(text)

# %%
feas = squad_convert_example_to_features2(squad_examples[0], 512, 128, 128, 'max_length', True, tk_nf)
# %%
type(pair_hold)
# %%
tk_nf.encode_plus(text_hold, pair_hold)
# %%
len(squad_examples[0].doc_tokens)
# %%
