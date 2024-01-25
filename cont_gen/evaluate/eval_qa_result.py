"""
Evaluate the output of span detection qa models.
"""
import json
import pandas as pd
from typing import List
from cont_gen.evaluate.token_metrics import compute_f1_iou_metrics
from cont_gen.evaluate.eval_genqa_result import eval_genqa_by_token_merge_doc

def convert_qa_pred_data(qa_pred_data, cla_list):
    """
    Convert all qa pred data to genqa pred data format
    """
    new_data = []
    for qaid, result in qa_pred_data.items():
        title, cla_name = qaid.split('__')
        q_id = cla_list.index(cla_name)
        if result['pred_text'] == "":
            pred_text = "None"
        else:
            pred_text = result['pred_text']
        new_data.append({
            'title': title,
            'q_id': q_id,
            'prediction': pred_text
        })
    return new_data

def eval_qa_by_token_merge_doc(chunk_data, pred_data, cla_list):
    """
    Evaluate baseline qa model by token level f1 and iou.

    Equivalent implementation of eval_genqa_result.eval_genqa_by_token_merge_doc 

    We keep a multi level dict from q_id to title to result of gold and pred
    """

    genqa_pred_data = convert_qa_pred_data(pred_data, cla_list)
    return eval_genqa_by_token_merge_doc(chunk_data, genqa_pred_data)

if __name__ == '__main__':
    import argparse
    import numpy as np
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_data', help = 'path of chunk data')
    parser.add_argument('cla_list', help = 'path of original clause names')
    parser.add_argument('pred_data', help = 'path of pred data')

    args = parser.parse_args()

    chunk_data = [json.loads(k) for k in open(args.chunk_data)]
    pred_data = pickle.load(open(args.pred_data, 'rb'))

    cla_list = json.load(open(args.cla_list))

    f1_by_cls, iou_by_cls = eval_qa_by_token_merge_doc(chunk_data, pred_data, cla_list)

    print(f'macro f1: {np.mean(f1_by_cls)}')
    print(f'macro IOU: {np.mean(iou_by_cls)}')
