"""
Check the token level metrics of genqa
"""
# %%
import json
import sys
import numpy as np
from typing import List
# %%
def load_jsonl(path):
    return [json.loads(k) for k in open(path)]
# %%
chunk_data = load_jsonl('../../data/cuad_clean/CUADv1_chunks_merge.jsonl')
pred_path = '../../runs/genqa/flan-t5-large_quest_lr1e-4_bs16/preds_ckpt_8361.jsonl'
pred_data = load_jsonl(pred_path)
all_titles = set([k['title'] for k in pred_data])
chunk_data = [k for k in chunk_data if k['title'] in all_titles]
print(f'title: {len(all_titles)}, chunks: {len(chunk_data)}')
# %%
cla_info = json.load(open('../../data/clause/clause_info.json'))
# %%
def parse_prediction(pred_str:str)->List[str]:
    if pred_str.startswith('None'):
        return []
    return pred_str.split('\n')
def get_compound_data(chunk_data, pred_data):
    """
    Merge chunk data and pred data.

    Return: map from title to clause results that is a list of each clause's result:
        q_id
        clause_name
        gold
        pred
    """
    comp_data = {}
    def add_title(title):
        if title in comp_data:
            return
        comp_data[title] = [
            {
                'q_id': i,
                'clause_name': cla_info[i]['category'],
                'gold': [],
                'pred': []
            } for i in range(len(cla_info))
        ]
        return
    
    for chunk in chunk_data:
        title = chunk['title']
        add_title(title)
        for qa in chunk['qas']:
            gold_text = [k['text'] for k in qa['answers']]
            comp_data[title][qa['q_id']]['gold'].extend(gold_text)
    
    for pred_d in pred_data:
        title = pred_d['title']
        pred_list = parse_prediction(pred_d['prediction'])
        if len(pred_list) == 0:
            continue
        add_title(title)
        comp_data[title][pred_d['q_id']]['pred'].extend(pred_list)
    
    return comp_data
# %%
comp_data = get_compound_data(chunk_data, pred_data)
# %%
sample_r = comp_data[list(all_titles)[1]]
# %%
for cla_r in sample_r:
    print(f'{cla_r["q_id"]} {cla_r["clause_name"]}')
    print(f'Gold: {" ".join(cla_r["gold"])}')
    print(f'Pred: {" ".join(cla_r["pred"])}')
    print('-'*20)
# %%
