"""Compute predictions of QA"""
# %%
import json
from pathlib import Path
from importlib import reload
from transformers import AutoTokenizer
import context
import numpy as np
import re
import pickle
from collections import defaultdict

import cont_gen.qa.compute_pred as qa_cp
# %%
reload(qa_cp)
from cont_gen.qa.compute_pred import compute_predictions_logits, compute_example_pred, propose_prelim_spans
from cont_gen.data_process.build_qa_feature import create_examples
from cont_gen.data_process.basic import DocTokens
# %%
def get_envs():
    from dotenv.main import dotenv_values
    envs = dotenv_values('../.env')
    return envs
envs = get_envs()
# %%
test_data = json.load(open(envs['CUAD_TEST']))['data']
# %%
features = pickle.load(open('../data/features/qa_roberta_test.pkl', 'rb'))
# %%
results = pickle.load(open('../runs/qa/roberta-base_lr1e-4_bs16/checkpoint-12000/model_outputs.pkl', 'rb'))
# %%
doc_objs = pickle.load(open('../data/doc/doc_tokens_roberta_rm.pkl', 'rb'))
# %%
id2doc = pickle.load(open('../data/doc/doc_id_text.pkl', 'rb'))
id2doc = {k['doc_id']: k['doc'] for k in id2doc}
doc_objs = [DocTokens(**k, doc_text = id2doc[k['doc_id']]) for k in doc_objs]
# %%

# %%
examples = create_examples(test_data)
# %%
example_preds = compute_predictions_logits(
    examples,
    all_features= features,
    all_results=results,
    doc_objs = doc_objs,
    n_best_size = 20,
    max_answer_length=30
)
# %%
# debug
doc_map = {k['doc_id']:k for k in doc_objs}
# get feature indexes of one example
exa_id_to_feat_ids = defaultdict(list)
for i, feature in enumerate(features):
    exa_id_to_feat_ids[feature['qa_id']].append(i)

# %%
example = examples[2216]
doc_obj = doc_map[example['doc_id']]

feature_indexes = exa_id_to_feat_ids[example['qa_id']]
e_feats = [features[k] for k in feature_indexes]
e_res = [results[k] for k in feature_indexes]
# %%
len(doc_obj.token_to_char)
# %%
for feat in e_feats:
    print(feat['span_start'], feat['context_offset'])
# %%
e_pred = qa_cp.compute_example_pred(feature_indexes, e_feats, e_res, doc_obj, 20, 30)
# %%
all_prelim_spans = []
for (feature_index, feature, result) in zip(feature_indexes, e_feats, e_res):
    prelim_spans = propose_prelim_spans(
        result['start_logits'], result['end_logits'], feature, 
        20, max_answer_length=30
    )
    for span in prelim_spans:
        span['feature_index'] = feature_index
        all_prelim_spans.append(span)
# %%
len(all_prelim_spans)
# %%
all_prelim_spans[0].keys()
# %%
end_indexes = [k['end_index'] for k in all_prelim_spans]
print(max(end_indexes))
# %%
doc_obj.token_to_char[1792 - 36 + max(end_indexes)]
# %%
len(doc_obj.token_to_char)
# %%
# look the pred
e_pred = example_preds[examples[100]['qa_id']]
# %%
