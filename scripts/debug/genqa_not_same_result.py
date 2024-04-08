"""
For genqa, the evaluation results of two runnings with same config but different code are different. Look into this Issue.

Detailed evaluation shows that the issue already shows after training, so it
may originate from data pre-processing, data loading and training.
"""
# %%
import json
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from functools import partial
import re
from collections import Counter
import difflib

from transformers import AutoTokenizer
# %%
def load_jsonl(f):
    return [json.loads(k) for k in open(f)]

def str_sim(a, b):
    """Compare the similarity of two string"""
    return difflib.SequenceMatcher(a = a, b = b).ratio()
# %%
# Load various data
clean_dir = Path('../../data/cuad_clean')
chunk_prev = load_jsonl(clean_dir / 'CUADv1_chunks_merge.jsonl')
# title, chunk_index, text, para_idx, para_offset, qas
prompt_prev = load_jsonl('../../data/cuad_prompts/train_prompts_quest.jsonl')
# title, chunk_index, q_id, prompt

para_new = load_jsonl(clean_dir / 'flan-t5_512/CUAD_paras.jsonl')
# title, paras
prompt_new = load_jsonl(clean_dir / 'flan-t5_512/genqa/quest_train_max-asw-len_80.jsonl')
# 'title', 'para_idx', 'q_id', 'source', 'target'
# %%
prompt_fix = load_jsonl(clean_dir / 'flan-t5_512/genqa/quest_train_max-asw-len_80_fix.jsonl')
# %%
# check processed data
tr_cache_new = pickle.load(open('../../cache/cached_quest_train_max-asw-len_80.jsonl_flan-t5-large_v1.0.pkl', 'rb'))
# %%
tk_flan = AutoTokenizer.from_pretrained('google/flan-t5-large')
# %%
sr_text = tk_flan.decode(tr_cache_new[0]['input_ids'])
tg_text = tk_flan.decode(tr_cache_new[0]['labels'])
# %%
tk_flan.tokenize('Hello \n May')
# %%
# Compare chunk data and para data.
# convert to flat structure of chunk and para data.
def chunk_to_df(chunk_data):
    return pd.DataFrame(chunk_data)

def para_to_df(para_data):
    para_dicts = []
    for para_d in para_data:
        title = para_d['title']
        for para_idx, para in enumerate(para_d['paras']):
            para_dicts.append({
                "title": title,
                "para_idx": para_idx,
                **para})
    return pd.DataFrame(para_dicts)
# %%
chunk_df = chunk_to_df(chunk_prev)
para_df = para_to_df(para_new)
# All these two contains all train and test data
# %%
titles = chunk_df['title'].unique().tolist()
print(f'num titles: {len(titles)}')
# %%
"""Filtering function"""
def has_qid(x, q_id):
    avi_qies = [k['q_id'] for k in x['qas']]
    return q_id in avi_qies

def get_pos(pmt_df):
    """Return positive sample indicators"""
    if 'prompt' in pmt_df.columns:
        flag = pmt_df['prompt'].apply(lambda k: not k.endswith('None'))
        return pmt_df[flag]
    else:
        return pmt_df[~pmt_df['target'].str.match('None')]

def has_multi_qid(x, q_id):
    """
    Return True if one para instance has multiple sample of clauses in qas
    """
    avi_qies = [k['q_id'] for k in x['qas']]
    return len([k for k in avi_qies if k == q_id]) > 1

def split_src_tgt(text):
    m = re.search(r'Answer:\n', text)
    pos = m.span(0)[1]
    return text[:pos], text[pos:]

# %%
sel_tt = titles[10]
cont_chunk = chunk_df[chunk_df['title'] == sel_tt]
cont_para = para_df[para_df['title'] == sel_tt]
# %%

# %%
sel_qid = 18
q_chunk = chunk_df[chunk_df.apply(partial(has_qid, q_id = sel_qid), axis = 1)]
q_para = para_df[para_df.apply(partial(has_qid, q_id = sel_qid), axis = 1)]
# %%
# Count clause paragraphs
q_id_para_ct = {} # each clause has how many paragraphs
q_id_cont_ct = {} # each clause exists how many contracts
for i in range(41):
    part = para_df[para_df.apply(partial(has_qid, q_id = i), axis = 1)]
    q_id_para_ct[i] = len(part)
    q_id_cont_ct[i] = part['title'].nunique()
# %%
{i:(q_id_cont_ct[i], q_id_para_ct[i]) for i in range(41)}
# %%
# Check prompts
pmt_prev_df = pd.DataFrame(prompt_prev)
pmt_new_df = pd.DataFrame(prompt_new)
# %%

# %%
# select q_id and title
sel_qid = 10
q_prev = get_pos(pmt_prev_df[pmt_prev_df['q_id'] == sel_qid])
q_new = get_pos(pmt_new_df[pmt_new_df['q_id'] == sel_qid])
# %%
part_titles = q_prev['title'].unique().tolist()
sel_tt = part_titles[0]
tq_prev = q_prev[q_prev['title'] == sel_tt]
tq_new = q_new[q_new['title'] == sel_tt]
# %%
prev_pstr = tq_prev.iloc[0]['prompt']
new_pstr = tq_new.iloc[0]['source'] + tq_new.iloc[0]['target']
print(prev_pstr)
print(new_pstr)
# %%
"""
Statistic the distribution of the bug, that one clause exists more than one time
in qas.
"""
cla_multi_count = {}
# map from clause type id to num of paras with 
# more than one of this clause
for i in range(41):
    part = chunk_df[chunk_df.apply(partial(has_multi_qid, q_id = i), axis = 1)]
    cla_multi_count[i] = len(part)
# Ovservation: only a small portion of paragraphs has duplicate clause types.
# %%
"""
Compare the prompt data: prompt_prev and prompt_fix
"""
print(f'num of prev prompt: {len(prompt_prev)}')
print(f'num of fixed prompt: {len(prompt_fix)}')
# filter positive samples
prompt_prev_pos = [k for k in prompt_prev if not k['prompt'].endswith('None')]
prompt_fix_pos = [k for k in prompt_fix if not k['target'].endswith('None')]
print(f'num pos prev prompt: {len(prompt_prev_pos)}')
print(f'num pos fixed prompt: {len(prompt_fix_pos)}')
# %%
# find matched titles
tr_tts = list(set([k['title'] for k in prompt_fix]))
miss_pmt_prev = {} # map from title to [q_id, chunk_index]
miss_pmt_new = {} # map from title to [q_id, para_index]

for title in tr_tts[1:2]:
    part_prev = [k for k in prompt_prev_pos if k['title'] == title]
    part_new = [k for k in prompt_fix_pos if k['title'] == title]

    match_pairs_idx = []
    part_prev_sources = [split_src_tgt(k['prompt'])[0] for k in part_prev]

    for ni, pt_new_d in enumerate(part_new):
        src = pt_new_d['source']
        if src in part_prev_sources:
            oi = part_prev_sources.index(src)
            old_tgt = split_src_tgt(part_prev[oi]['prompt'])[1]
            sim_score = str_sim(old_tgt, pt_new_d['target'])
            if sim_score> 0.9:
                match_pairs_idx.append((oi, ni))
            else:
                print(ni, oi, sim_score)
    
    print(len(match_pairs_idx))

    match_new_ids = [k[1] for k in match_pairs_idx]
    miss_new_ids = [k for k in range(len(part_new)) if k not in match_new_ids]
# %%
print(part_new[3]['target'])
# %%
print(split_src_tgt(part_prev[2]['prompt'])[1])
# %%
"""
Check the distribution of negative sampling.
"""
prompt_prev_neg = [k for k in prompt_prev if k['prompt'].endswith('None')]
prompt_fix_neg = [k for k in prompt_fix if k['target'].endswith('None')]
tt_ct_prev = Counter([k['title'] for k in prompt_prev_neg]) 
tt_ct_fix = Counter([k['title'] for k in prompt_fix_neg])
# %%
for i, tt in enumerate(tr_tts):
    print(f'{i}: {tt_ct_prev[tt]} {tt_ct_fix[tt]}')
# %%
