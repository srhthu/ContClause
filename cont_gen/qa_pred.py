"""
Given qa results, get predictions.
"""
import json
from pathlib import Path
import argparse

import numpy as np
import re
import pickle
from collections import defaultdict

from cont_gen.qa.compute_pred import compute_predictions_logits
from cont_gen.data_process.basic import DocTokens
from cont_gen.data_process.build_qa_feature import create_examples

def get_envs():
    from dotenv.main import dotenv_values
    envs = dotenv_values('.env')
    return envs


def main():
    envs = get_envs()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_outputs', help = 'path or folder that contain model outputs')
    parser.add_argument('--max_answer_length', type = int, default = 256)
    parser.add_argument('--features', default = 'data/features/qa_roberta_test.pkl')
    parser.add_argument('--test_data', default = envs['CUAD_TEST'])
    parser.add_argument('--doc_tokens', default = 'data/doc/doc_tokens_roberta_rm.pkl')
    parser.add_argument('--doc_text', default = 'data/doc/doc_id_text.pkl')

    args = parser.parse_args()

    print('Load test data')
    test_data = json.load(open(args.test_data))['data']

    id2doc = {
        k['title']: k['paragraphs'][0]['context']
            for k in test_data
    }

    print('Load documents')
    doc_objs = pickle.load(open(args.doc_tokens, 'rb'))
    doc_objs_map = {k['doc_id']: k for k in doc_objs}

    doc_objs = [DocTokens(**doc_objs_map[k], doc_text = doc) for k,doc in id2doc.items()]

    print('Load test features')
    features = pickle.load(open(args.features, 'rb'))

    print('Load test results')
    pred_sufix = f'ml{args.max_answer_length}'
    if Path(args.model_outputs).is_file():
        res_path = args.model_outputs
        save_path = Path(args.model_outputs).parent / f'predictions_{pred_sufix}.pkl'
    elif Path(args.model_outputs).is_dir():
        res_path = Path(args.model_outputs) / 'model_outputs.pkl'
        save_path = Path(args.model_outputs) / f'predictions_{pred_sufix}.pkl'
    else:
        raise ValueError(f'wrong path: {args.model_outputs}')
    results = pickle.load(open(res_path, 'rb'))

    examples = create_examples(test_data)
    
    example_preds = compute_predictions_logits(
        examples,
        all_features= features,
        all_results=results,
        doc_objs = doc_objs,
        n_best_size = 20,
        max_answer_length = 256
    )

    with open(save_path, 'wb') as f:
        pickle.dump(example_preds, f)

if __name__ == '__main__':
    main()