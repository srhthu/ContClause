"""Check the clean data"""
# %%
import json
from pathlib import Path
import sys
import re
sys.path.insert(0, str(Path(__file__).parents[1].absolute()))
# %%
import numpy as np
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import string
from importlib import reload
import cont_gen
reload(cont_gen)
from cont_gen.data_process.utils import (
    cut_spans_return_offset, 
    remove_spans_and_return_mapping,
    merge_spans,
    span_contain,
    relocate_spans_into_range,
    reverse_char_map,
    group_by
)
# %%
with open('../data/cuad_split/CUADv1.json') as f:
    ori_data = json.load(f)

with open('../data/cuad_clean/CUADv1.jsonl') as f:
    clean_data = [json.loads(k) for k in f]
# %%
with open('../data/cuad_clean/CUADv1_paras.jsonl') as f:
    paras_data = [json.loads(k) for k in f]    
with open('../data/cuad_clean/CUADv1_paras_merge.jsonl') as f:
    paras_merge_data = [json.loads(k) for k in f] 

# %%
with open('../data/cuad_clean/CUADv1_chunks_merge.jsonl') as f:
    chunk_data = [json.loads(k) for k in f] 

# %%
ori_sample = ori_data['data'][0]
cln_sample = clean_data[0]
# %%
print(ori_sample['paragraphs'][0]['context'][:200])
print(cln_sample['doc_text'][:200])
# %%
for d in clean_data:
    for qa in d['qas']:
        for answer in qa['answers']:
            assert answer['text'] == d['doc_text'][answer['start_pos']: answer['end_pos'] + 1]
# %%
ori_qas = [k for k in ori_sample['paragraphs'][0]['qas'] if not k['is_impossible']]
cln_qas = [k for k in cln_sample['qas'] if not k['is_impossible']]
# %%
# Show clean data
for ori_qa, cln_qa in zip(ori_qas, cln_qas):
    print('=='*20)
    print('Original')
    for answer in ori_qa['answers']:
        print('{}\n{}'.format(answer['answer_start'], answer['text']))
    print('Clean')
    for answer in cln_qa['answers']:
        print('{} - {}\n{}'.format(answer['start_pos'], answer['end_pos'], answer['text']))

# %%
non_ascii = set([k for k in cln_sample['doc_text'] if ord(k) > 127])
print(len(non_ascii))
# %%
# print non-ascii characters
non_ascii = []
for d in clean_data:
    non_ascii.extend([k for k in d['doc_text'] if ord(k) > 127])

ct = Counter(non_ascii)
print(len(ct))
# %%
non_print = []
for d in clean_data:
    non_print.extend([k for k in d['doc_text'] if k not in string.printable])
ct = Counter(non_print)
print(len(ct))
# %%
# Investigate the paragraph lenght.
p_lens = []
short_paras = []
for i, d in enumerate(clean_data):
    for pi, para in enumerate(d['doc_text'].split('\n')):
        if len(para) > 0:
            p_lens.append(len(para))
            if len(para) <= 10:
                short_paras.append((i, pi, para))           
# %%
_ = plt.hist(p_lens, bins = 50, range = (0, 2000))
# %%
short_paras[0]
# %%
print(clean_data[7]['doc_text'].split('\n')[1:4])
# %%
print(ori_data['data'][7]['paragraphs'][0]['context'][76:200])
# %%
ori_data['data'][7]['paragraphs'][0]['qas'][0]
# %%
part = [k for k in short_paras if len(k[2]) <= 10]
# print(set([k[2] for k in part]))
print(len(part))
# %%
# Check para data to find dist of paras with answers
para_with_ans = []
for ci, d in enumerate(paras_data):
    for pi, para in enumerate(d['paras']):
        if len(para['qas']) > 0:
            para_with_ans.append((ci, pi, len(para['text'])))
# %%
plt.hist([k[2] for k in para_with_ans], bins=100, range=(0,100))
# %%
part = [k for k in para_with_ans if k[2] <= 10]
print(len(part))
# %%
paras_data[93]['paras'][5]
# %%
# check merged paragraphs
for d in paras_merge_data:
    for para in d['paras']:
        para_text = para['text']
        for qa in para['qas']:
            for asw in qa['answers']:
                st, ed = asw['start_pos'], asw['end_pos']
                assert asw['text'] == para_text[st:ed+1]
# %%
# compare para numbers after merge short paras
ave_n_para = np.mean([len(d['paras']) for d in paras_data])
ave_n_para_merge = np.mean([len(d['paras']) for d in paras_merge_data])
print(ave_n_para)
print(ave_n_para_merge)
ave_p_len = np.mean([len(p['text']) for d in paras_data for p in d['paras']])
ave_p_len_merge = np.mean([len(p['text']) for d in paras_merge_data for p in d['paras']])
print(ave_p_len)
print(ave_p_len_merge)
# %%
# [TODO] check the chunk data
# %%
# Find spans that are too close and should be merged
all_span_dist = []
for idx, sample in enumerate(clean_data):
    for qidx, qa in enumerate(sample['qas']):
        span_dist = [(
            qa['answers'][i]['start_pos'] 
            - qa['answers'][i-1]['end_pos']
            ) for i in range(1, len(qa['answers']))]
        if len(span_dist) > 0:
            all_span_dist.append((idx, qidx, span_dist))
print('Total qas:', len([qa for sample in clean_data for qa in sample['qas'] if not qa['is_impossible']]))
print('total span_dist', len(all_span_dist))
# %%
_ = plt.hist([min(k[2]) for k in all_span_dist], bins = 50, range = (0, 100))
# %%
# find annotations with span dist == 2
part_span_dist = [k for k in all_span_dist if min(k[2]) == 6]
print(len(part_span_dist))
# %%
t = part_span_dist[2]
for ele in clean_data[t[0]]['qas'][t[1]]['answers']:
    print(ele)
# %%
# get key value pairs
def get_clause_values(sample):
    d = OrderedDict()
    for qa in sample['qas']:
        if qa['is_impossible']:
            continue
        cla_name = qa['qa_id'].split('__')[-1]
        anss = [k['text'] for k in qa['answers']]
        d[cla_name] = anss
    return d
# %%
cla_values = get_clause_values(clean_data[1])
# %%
i = 0
for k,v in cla_values.items():
    print(f'[{i+1}] {k}: {v[0]}')
    i += 1
# %%
# check questions
quests = [k['question'] for k in ori_data['data'][0]['paragraphs'][0]['qas']]
# %%
