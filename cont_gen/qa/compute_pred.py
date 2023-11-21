"""Get model predictions given model outputs"""

import json
import pickle
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any, Union
from tqdm import tqdm
import traceback

from cont_gen.data_process.basic import (
    DocTokens, QA_Example, QA_Feature
)

def get_best_indexes(logits, n_best_size):
    # sort in descending order
    rank = np.flip(np.argsort(logits))
    return rank[:n_best_size].tolist()

def propose_prelim_spans(
    start_logits, 
    end_logits, 
    feature: QA_Feature,
    n_best_size, 
    max_answer_length,
):
    """
    Propose possible combination of span start and end position and filter 
    """
    start_indexes = get_best_indexes(start_logits, n_best_size)
    end_indexes = get_best_indexes(end_logits, n_best_size)

    for start_i in start_indexes:
        for end_i in end_indexes:
            # span should in the context window, not question, pad or cls
            ctx_end = feature['context_offset'] + feature['paragraph_len'] - 1
            if start_i < feature['context_offset'] or start_i > ctx_end:
                continue
            if end_i < feature['context_offset'] or end_i > ctx_end:
                continue

            # check start token has the max context
            # [TODO]

            if end_i < start_i:
                continue

            if (end_i  - start_i + 1) > max_answer_length:
                continue

            yield {
                ''
                'start_index': start_i,
                'end_index': end_i,
                'start_logit': start_logits[start_i],
                'end_logit': end_logits[end_i],
            }


def get_final_prediction(prelim_pred, feature, doc_obj):
    """Map the start/end index to char and token position"""
    pred = prelim_pred
    if pred['start_index'] > 0:  # this is a non-null prediction
        doc_offset = - feature['context_offset'] + feature['span_start']
        token_start = pred['start_index'] + doc_offset
        token_end = pred['end_index'] + doc_offset
        char_start = doc_obj['token_to_char'][token_start][0]
        char_end = doc_obj['token_to_char'][token_end][1] - 1
        final_text = doc_obj['doc_text'][char_start: char_end + 1]
    else:
        token_start = token_end = char_start = char_end = -1
        final_text = ""

    final_pred = {
        'text': final_text,
        'start_logit': pred['start_logit'],
        'end_logit': pred['end_logit'],
        'char_start': char_start,
        'char_end': char_end,
        'token_start': token_start,
        'token_end': token_end,
        'qa_id': feature['qa_id'],
        'feature_index': pred['feature_index']
    }
    return final_pred

def compute_softmax(scores):
    # numerical stability
    if len(scores) == 0:
        return []
    scores = np.array(scores)
    max_score = np.max(scores)
    e_scores = np.exp(scores - max_score)
    return e_scores / e_scores.sum()

def _get_score(k):
        return k['start_logit'] + k['end_logit']

def compute_example_pred(
    feature_indexes, features, results, 
    doc_obj,
    n_best_size, max_answer_length
):
    """Compute one example's predictions"""
    feat_map = {i:k for i,k in zip(feature_indexes, features)}

    # Propose possible spans with combinations of start and end indexes
    all_prelim_spans = []
    for (feature_index, feature, result) in zip(feature_indexes, features, results):
        prelim_spans = propose_prelim_spans(
            result['start_logits'], result['end_logits'], feature, 
            n_best_size, max_answer_length=max_answer_length
        )
        for span in prelim_spans:
            span['feature_index'] = feature_index
            all_prelim_spans.append(span)
    
    # Add the minimum score of null span
    feat_null_score = [
        result['start_logits'][0] + result['end_logits'][0] 
        for result in results
    ]
    min_null_index = np.argmin(feat_null_score)
    min_null_result = results[min_null_index]
    null_prelim_span = {
        'feature_index': feature_indexes[min_null_index], # this index is the global index
        'start_index': 0,
        'end_index': 0,
        'start_logit': min_null_result['start_logits'][0],
        'end_logit': min_null_result['end_logits'][0],
    }
    all_prelim_spans.append(null_prelim_span)

    # sort all preliminary spans
    all_prelim_spans = sorted(
        all_prelim_spans, 
        key=lambda x: (x['start_logit'] + x['end_logit']), 
        reverse=True
    )

    # keep top predictions and remove repeated ones
    nbest = []
    seen_char_positions = set()
    for pred in all_prelim_spans:
        if len(nbest) >= n_best_size:
            break
        feature = feat_map[pred['feature_index']]
        final_pred = get_final_prediction(pred, feature, doc_obj)

        char_pos = (final_pred['char_start'], final_pred['char_end'])
        if char_pos in seen_char_positions:
            continue
        seen_char_positions.add(char_pos)

        nbest.append(final_pred)

    # if we didn't include the empty option in the n-best, include it
    if(-1, -1) not in seen_char_positions:
        null_final_pred = get_final_prediction(
            null_prelim_span, 
            feat_map[null_prelim_span['feature_index']], 
            doc_obj
        )
        nbest.append(null_final_pred)

    # calculate probabilities of all predicted spans
    total_scores = [k['start_logit'] + k['end_logit'] for k in nbest]
    probs = compute_softmax(total_scores)
    for i in range(len(nbest)):
        nbest[i]['prob'] = probs[i]
    
    # Find the best span, it should have higher score than null span
    score_null = _get_score(null_prelim_span)
    best_score = _get_score(nbest[0])
    if score_null - best_score < 0.0:
        pred_text = nbest[0]['text']
    else:
        pred_text = ""

    example_pred = {
        'all_preds': nbest,
        'score_null': score_null,
        'pred_text': pred_text
    }
    return example_pred

def compute_predictions_logits(
    all_examples: List[DocTokens],
    all_features: List[Union[Dict, QA_Feature]],
    all_results: List[Dict[str, Any]],
    doc_objs: List[DocTokens],
    n_best_size,
    max_answer_length,
):
    """
    Return example predictions with scores.
    """
    # prepare doc map
    doc_map = {k['doc_id']:k for k in doc_objs}

    # get feature indexes of one example
    exa_id_to_feat_ids = defaultdict(list)
    for i, feature in enumerate(all_features):
        exa_id_to_feat_ids[feature['qa_id']].append(i)

    example_preds = {}
    for (example_index, example) in tqdm(list(enumerate(all_examples))):
        # Process one example with its features
        doc_obj = doc_map[example['doc_id']]

        feature_indexes = exa_id_to_feat_ids[example['qa_id']]
        features = [all_features[k] for k in feature_indexes]
        results = [all_results[k] for k in feature_indexes]

        assert features[0]['example_index'] == example_index
        # qa_id and example_index are equivalent to identify the example

        try:
            example_pred = compute_example_pred(
                feature_indexes, features, results, doc_obj, 
                n_best_size=n_best_size, 
                max_answer_length=max_answer_length
            )
        except:
            print(traceback.format_exc())
            print('Error index:', example_index)
            return None

        example_preds[example['qa_id']] = example_pred

    return example_preds

