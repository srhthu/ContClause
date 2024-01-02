"""Check the clean data"""
# %%
import json
from pathlib import Path
import sys
import re
sys.path.insert(0, str(Path(__file__).parents[1].absolute()))
# %%
from collections import Counter
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

with open('../data/cuad_clean/CUADv1.json') as f:
    clean_data = [json.loads(k) for k in f]
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
for ori_qa, cln_qa in zip(ori_qas, cln_qas):
    print('=='*20)
    print('Original')
    for answer in ori_qa['answers']:
        print('{}\n{}'.format(answer['answer_start'], answer['text']))
    print('Clean')
    for answer in cln_qa['answers']:
        print('{} - {}\n{}'.format(answer['start_pos'], answer['end_pos'], answer['text']))
# %%
doc = ori_sample['paragraphs'][0]['context']
pat_sp = r' ([ ]+)'
pat_nl = r'\n([\n]+)'

cut_spans = []
for pat in [pat_sp, pat_nl]:
    for m in re.finditer(pat, doc):
        cut_spans.append(m.span(1))
# %%
text_parts, offsets = cut_spans_return_offset(doc, cut_spans)
# %%
for i in range(1, len(cut_spans)):
    if cut_spans[i][0] < cut_spans[i-1][1]:
        print(i)
        break
# %%
non_ascii = set([k for k in cln_sample['doc_text'] if ord(k) > 127])
print(len(non_ascii))
# %%
non_ascii = []
for d in clean_data:
    non_ascii.extend([k for k in cln_sample['doc_text'] if ord(k) > 127])

ct = Counter(non_ascii)
print(len(ct))
# %%
non_print = []
for d in clean_data:
    non_print.extend([k for k in cln_sample['doc_text'] if k not in string.printable])

ct = Counter(non_print)
print(len(ct))
# %%
