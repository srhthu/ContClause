"""Given predictions, calculate metrics"""

import collections
import json
import math
import re
import string
import json

# Copy from cuad project
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

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return [int(gold_toks == pred_toks)] * 3
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions.

    Return:
        exact_scores: map from qa_id to (0,1)
        f1_scores: map from qa_id to f1 score
    """
    exact_scores = {}
    f1_scores = {}
    prec_scores = {}
    recall_scores = {}

    for example in examples:
        qas_id = example['qa_id']
        gold_answers = [normalize_answer(t) for t in example['answer_texts']]
        gold_answers = list(filter(lambda k:k, gold_answers))

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]['pred_text']
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        # Change to a practical F1
        # f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)
        
        prec, recall, f1 = compute_f1(' '.join(gold_answers), prediction)
        f1_scores[qas_id] = f1
        prec_scores[qas_id] = prec
        recall_scores[qas_id] = recall


    return exact_scores, f1_scores, prec_scores, recall_scores

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    """
    If the no_answer prob bigger than the threshold, the pred is thought to be null and 
    change the score based on whether it is truly null answer
    """
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores

def mean_score(score_map, qid_list = None):
    s_list = list(score_map.values()) if qid_list is None \
                else [score_map[k] for k in qid_list]
    total = len(s_list)
    ave_s = 100 * sum(s_list) / total
    return ave_s, total

def make_eval_dict(exact_scores, f1_scores, prefix = '', qid_list=None):
    exact, total = mean_score(exact_scores, qid_list)
    f1, _ = mean_score(f1_scores, qid_list)
    metrics = {'exact': exact, 'f1': f1, 'total': total}
    return {f'{prefix}{k}':v for k,v in metrics.items()}

def squad_evaluate(
    examples, 
    preds, 
    no_answer_probs=None, 
    no_answer_probability_threshold=1.0
):
    qas_id_to_has_answer = {e['qa_id']: not e['is_impossible'] for e in examples}
    has_answer_qids = [qas_id for qas_id, psb in qas_id_to_has_answer.items() if psb]
    no_answer_qids = [qas_id for qas_id, psb in qas_id_to_has_answer.items() if not psb]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1, prec, recal = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(
        f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )

    # evaluation = make_eval_dict(exact_threshold, f1_threshold)
    evaluation = {
        'exact': mean_score(exact)[0],
        'f1': mean_score(f1)[0],
        'p': mean_score(prec)[0],
        'r': mean_score(recal)[0],
    }

    if has_answer_qids:
        has_ans_eval = make_eval_dict(
            exact_threshold, f1_threshold, qid_list=has_answer_qids, prefix = 'HasAns_'
        )
        evaluation.update(has_ans_eval)

    if no_answer_qids:
        no_ans_eval = make_eval_dict(
            exact_threshold, f1_threshold, qid_list=no_answer_qids, prefix = "NoAns_"
        )
        evaluation.update(no_ans_eval)

    return evaluation