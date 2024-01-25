"""
Evaluate the output of generative qa models.
"""
import json
import pandas as pd
from typing import List
from cont_gen.evaluate.token_metrics import compute_f1_iou_metrics

def parse_prediction(pred_str:str)->List[str]:
    if pred_str.startswith('None'):
        return []
    return pred_str.split('\n')

def get_title_qid_from():
    ...

def eval_genqa_by_token_merge_doc(chunk_data, pred_data):
    """
    Evaluate generated spans by token level f1 and iou.

    For each document and a question, we collect all predictions and calculate metrics with all golden spans.
    
    This is only a simple implementation.

    Args:
        chunk_data: each sample is a dict of 
            title, chunk_index, text, para_idx, para_offset, qas
        pred_data: each sample is a dict of 
            title, chunk_index, q_id, prediction
    """
    # filter the test part of chunk data
    all_titles = set([k['title'] for k in pred_data])
    chunk_data = [k for k in chunk_data if k['title'] in all_titles]

    # Merge the answer of one title and q_id
    title_qid_to_answer = {}
    for chunk in chunk_data:
        for qa in chunk['qas']:
            key = (chunk['title'], qa['q_id'])
            if key not in title_qid_to_answer:
                title_qid_to_answer[key] = {
                    'gold': [],
                    'pred': [],
                }
            
            gold_text = [k['text'] for k in qa['answers']]

            title_qid_to_answer[key]['gold'].extend(gold_text)
    
    for pred_d in pred_data:
        key = (pred_d['title'], pred_d['q_id'])
        pred_list = parse_prediction(pred_d['prediction'])
        if len(pred_list) == 0:
            continue

        if key not in title_qid_to_answer:
            title_qid_to_answer[key] = {
                'gold': [],
                'pred': [],
            }
        title_qid_to_answer[key]['pred'].extend(pred_list)
    
    metric_records = []
    for (title, qid), answer in title_qid_to_answer.items():
        # if len(answer['gold']) == 0 and len(answer['pred']) == 0:

        metrics = compute_f1_iou_metrics(
            ' '.join(answer['gold']),
            ' '.join(answer['pred'])
        )
        metric_records.append({
            'title': title,
            'q_id': qid,
            **metrics
        })
    metric_df = pd.DataFrame(metric_records)

    macro_f1 = [metric_df[metric_df['q_id'] == i]['f1'].mean() for i in range(metric_df['q_id'].max() + 1)]
    macro_iou = [metric_df[metric_df['q_id'] == i]['iou'].mean() for i in range(metric_df['q_id'].max() + 1)]

    return macro_f1, macro_iou

if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_data', help = 'path of chunk data')
    parser.add_argument('pred_data', help = 'path of pred data')

    args = parser.parse_args()

    chunk_data = [json.loads(k) for k in open(args.chunk_data)]
    pred_data = [json.loads(k) for k in open(args.pred_data)]

    f1_by_cls, iou_by_cls = eval_genqa_by_token_merge_doc(chunk_data, pred_data)

    print(f'macro f1: {np.mean(f1_by_cls)}')
    print(f'macro IOU: {np.mean(iou_by_cls)}')
