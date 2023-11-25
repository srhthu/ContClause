"""Compute predictions of QA"""
# %%
import json
from pathlib import Path
from importlib import reload
import context
import numpy as np
import re
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple

from cont_gen.data_process.build_qa_feature import create_examples
import cont_gen.qa.eval as qa_eval
# %%
reload(qa_eval)
from cont_gen.qa.eval import squad_evaluate, compute_f1
# %%
def get_envs():
    from dotenv.main import dotenv_values
    envs = dotenv_values('../.env')
    return envs
envs = get_envs()
# %%
train_data = json.load(open(envs['CUAD_TRAIN']))['data']
tr_examples = create_examples(train_data)
# %%
test_data = json.load(open(envs['CUAD_TEST']))['data']
examples = create_examples(test_data)
# %%
pred_path = '../runs/qa/roberta-base_lr1e-4_bs16/checkpoint-12000/predictions_ml256.pkl'
preds = pickle.load(open(pred_path, 'rb'))

# %%
squad_evaluate(examples, preds)
# %%
has_anwser_idx = [i for i,e in enumerate(examples) if not e['is_impossible']]
null_anwser_idx = [i for i,e in enumerate(examples) if e['is_impossible']]
# %%
exa = examples[7]
e_preds = preds[exa['qa_id']]
print(exa['answer_texts'])
print(e_preds['pred_text'])
# %%
aa_idx = [eid for eid in has_anwser_idx if preds[examples[eid]['qa_id']]['pred_text'] != '']
# %%
tr_feats = pickle.load(open('../data/features/qa_roberta_train.pkl', 'rb'))
test_feats = pickle.load(open('../data/features/qa_roberta_test.pkl', 'rb'))
# %%
count_null = np.array([[0,0], [0,0]])
for e in examples:
    pred_text = preds[e['qa_id']]['pred_text']
    if not e['is_impossible']:
        if pred_text == '':
            count_null[0][1] += 1
        else:
            count_null[0][0] += 1
    else:
        if pred_text == '':
            count_null[1][1] += 1
        else:
            count_null[1][0] += 1

# %%
# save span info
pred_spans: List[Dict[str, List[Tuple[int, int]]]] = []
for cont_data in test_data:
    type2spans = {}
    for i, qas in enumerate(cont_data['paragraphs'][0]['qas']):
        qa_id = qas['id']
        e_pred = preds[qa_id]
        if e_pred['pred_text'] != '':
            span_info = e_pred['all_preds'][0]
            rg = (span_info['char_start'],  span_info['char_end'] + 1)
            if f'pred_{i}' not in type2spans:
                type2spans[f'pred_{i}'] = [rg]
            else:
                type2spans[f'pred_{i}'].append(rg)
    pred_spans.append(type2spans)
# %%
with open('../data/test_pred_spans.pkl', 'wb') as f:
    pickle.dump(pred_spans, f)
# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
im = ax.imshow(count_null, cmap="YlGn")
ax.set_xticks([0, 1], labels=['Has\nAns', 'Null'], fontsize=20)
ax.set_yticks([0, 1], labels=['Has\nAns', 'Null'], fontsize=20)
ax.set_ylabel('Gold', fontsize=20)
ax.set_xlabel('Predict', fontsize=20)
for i in range(2):
    for j in range(2):
        ax.text(j, i, count_null[i,j], ha="center", va="center", color="black", fontsize=20)
# %%
