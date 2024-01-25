"""
For CUAD task, evaluate the token levle f1 score.
"""
import collections
import json
import math
import re
import string
import json

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1_iou_metrics(a_gold, a_pred):
    """
    Given gold and pred text, compute metrics
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    # if len(gold_toks) == 0 or len(pred_toks) == 0:
    #     # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    #     return [int(gold_toks == pred_toks)] * 3

    if len(pred_toks) > 0:
        precision = 1.0 * num_same / len(pred_toks)
    else:
        # No pred but has gold (false negative)
        precision = 0.0
    if len(gold_toks) > 0:
        recall = 1.0 * num_same / len(gold_toks)
    else:
        # Has pred but no gold (false positive)
        recall = 0.
    if (precision + recall) > 1e-6:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.
    if num_same == 0:
        iou = 0.
    else:
        iou = num_same / (len(pred_toks) + len(gold_toks) - num_same)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'num_joint': num_same,
        'num_gold': len(gold_toks),
        'num_pred': len(pred_toks)
    }
    return metrics