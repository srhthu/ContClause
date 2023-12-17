"""Test the GenQA data process pipeline"""
# %%
import json
from pathlib import Path
from importlib import reload
from transformers import AutoTokenizer
import context
import numpy as np
import re
import pickle
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

from cont_gen.data_process import build_genqa_feature
from cont_gen.data_process import utils as dp_utils
# %%
reload(dp_utils)
reload(build_genqa_feature)
from cont_gen.data_process.utils import merge_spans, get_doc_with_spans, reverse_char_map
from cont_gen.data_process.build_genqa_feature import (
    split_paragraphs,
    remove_extra_space,
    get_genqa_examples
)
# %%
def get_envs():
    from dotenv.main import dotenv_values
    envs = dotenv_values('../.env')
    return envs
def load_json(path):
    with open(path, encoding='utf8') as f:
        return json.load(f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
# %%
# %%
envs = get_envs()
data_dir = Path('/home/shuirh/research/baselines/cuad/data')
cuad_v1 = load_json(data_dir / 'CUADv1.json')['data']
train_dt = load_json(data_dir / 'train_separate_questions.json')['data']
test_dt = load_json(data_dir / 'test.json')['data']
# %%
doc = train_dt[0]['paragraphs'][0]['context']
nat_paras = split_paragraphs(doc)
# %%
# check the function of remove_extra_space
p_text = nat_paras[10]['p_text']
# p_text = '  '
clean_text, char_map = remove_extra_space(p_text)
print(clean_text)
assert len(clean_text) == len(char_map)
for i,j in enumerate(char_map):
    assert clean_text[i] == p_text[j], (i,j)
# %%
# check merge spans
cla_names = [q['id'].split("__")[1] for q in cuad_v1[0]['paragraphs'][0]['qas']]
sim_dataset = get_doc_with_spans(cuad_v1)
# %%
ol_count = {} # overlap count
ol_rec = []
for i, data in enumerate(sim_dataset):
    for cla_i, spans in enumerate(data['all_answers']):
        rank_spans = merge_spans(spans)
        if len(spans) != len(rank_spans):
            if cla_i not in ol_count:
                ol_count[cla_i] = 0
            ol_count[cla_i] += 1
            ol_rec.append([i, cla_names[cla_i], spans, rank_spans])
print(ol_count)
# %%
part = [k for k in ol_rec if 'Rofr/Rofo/Rofn' == k[1]]
for k in part:
    for e in k:
        print(e)

# %%
# test gen_qa examples
gqa_examples = get_genqa_examples(cuad_v1)

# %%
# count paragraph tokens and check tokenize results
train_examples = load_pickle('../data/genqa/train_examples.pkl')
train_tok_r = load_pickle('../data/genqa/train_token_ids_llama2.pkl')
# %%
long_paras = [k for k in train_tok_r if len(k['token_ids']) > 1000]
print(len(long_paras))
# %%
# check span distance
all_dis = []
for i, data in enumerate(sim_dataset):
    for cla_i, spans in enumerate(data['all_answers']):
        rank_spans = merge_spans(spans)
        dis = [rank_spans[i][0] - rank_spans[i-1][1] for i in range(1, len(rank_spans))]
        all_dis.extend(dis)
# %%
print(len(all_dis))
# %%
print(sum([1 for k in all_dis if k == 0]))
# %%
dis_sep = defaultdict(list)
for i, data in enumerate(sim_dataset):
    for cla_i, spans in enumerate(data['all_answers']):
        rank_spans = merge_spans(spans)
        for i in range(1, len(rank_spans)):
            sep = data['doc'][rank_spans[i-1][1]:rank_spans[i][0]]
            dis_sep[len(sep)].append(sep)
# %%
Counter(dis_sep[7]).most_common(20)
# %%
dis_ct = list(Counter([k if k < 20 else 100 for k in all_dis if k < 20]).items())
x = [k[0] for k in dis_ct]
y = [k[1] for k in dis_ct]
plt.scatter(x, y)
