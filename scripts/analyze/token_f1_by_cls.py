"""
Analyze the token level f1 and IOU score across classes
"""
# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# %%
import pandas as pd
from importlib import reload
import cont_gen
from cont_gen.utils import load_json, load_jsonl
from cont_gen.evaluate.token_metrics import compute_f1_iou_metrics
# from cont_gen.evaluate.eval_genqa_result import 
# %%
cla_names = load_json('../../data/clause/ori_clause_names.json')
# %%
chunk_data = load_jsonl('../../data/cuad_clean/CUADv1_chunks_merge.jsonl')
# %%
preds = load_jsonl('../../runs/genqa/flan-t5-large_quest_desc_lr1e-4_bs16/preds_ckpt_6270.jsonl')
# %%
gold_data = []
for ck in chunk_data:
    for qa in ck['qas']:
        gold_text = [k['text'] for k in qa['answers']]
        gold_data.append({
            "title": ck['title'],
            "q_id": qa['q_id'],
            "gold_answer": gold_text
        })
gold_df = pd.DataFrame(gold_data)
# %%
pred_df = pd.DataFrame(preds)[['title', 'q_id', 'prediction']].rename(
    {'prediction': 'pred_answer'}, axis=1)
pred_df = pred_df[~pred_df['pred_answer'].str.startswith('None')]
pred_df['pred_answer'] = pred_df['pred_answer'].apply(lambda k: k.split('\n'))
pred_df.reset_index(inplace = True, drop = True)
pred_df = pred_df.groupby(['title', 'q_id'])['pred_answer'].apply(
    lambda l: [a for ele in l for a in ele]
).reset_index()
print(len(pred_df))
# %%
titles = set(pred_df['title'].unique().tolist())
test_df = gold_df[gold_df['title'].apply(lambda k: k in titles)]
print(len(test_df))
# %%
has_ellipsis = pred_df[pred_df['pred_answer'].apply(lambda k: any(['...' in a for a in k]))]
print(len(has_ellipsis))
# %%
comb_df = test_df.merge(pred_df, on = ['title', 'q_id'], how = 'outer')
for key in ['gold_answer', 'pred_answer']:
    comb_df[key] = comb_df[key].fillna("").apply(list)
print(len(comb_df))
# %%
# calculate metrics by clause type
def metric_fn(row):
    metrics_dt = compute_f1_iou_metrics(
        ' '.join(row['gold_answer']), ' '.join(row['pred_answer'])
    )
    # row = row.update(metrics_dt)
    return pd.Series(metrics_dt)
metric_df = comb_df.join(comb_df.apply(metric_fn, axis = 1))
# %%
metric_by_cls = metric_df.groupby('q_id')[['precision', 'recall', 'f1', 'iou']].apply(
    lambda df: df.mean(axis = 0)
).reset_index()
# %%
metric_by_cls['clause'] = metric_by_cls['q_id'].apply(lambda k: cla_names[k])
metric_by_cls = metric_by_cls.sort_values(by = 'f1', ascending = False)
# %%
with open('../../runs/genqa/flan-t5-large_quest_desc_lr1e-4_bs16/metrics_by_cls_6270.txt', 'w') as f:
    f.write(metric_by_cls.to_string())
# %%
