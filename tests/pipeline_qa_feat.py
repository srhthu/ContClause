"""
Test the pipeline of building qa features.
"""
# %%
import json
from pathlib import Path
from importlib import reload
from transformers import AutoTokenizer
import context
import numpy as np
import re
import pickle

from cont_gen.data_process import build_qa_feature
from cont_gen.data_process.basic import DocTokens
# %%
# reload the module if changed
reload(build_qa_feature)
from cont_gen.data_process.build_qa_feature import (
    create_examples, convert_features, convert_token_char_map,
    process_features
)
# %%
id2doc = pickle.load(open('../data/doc/doc_id_text.pkl', 'rb'))
id2doc = {k['doc_id']: k['doc'] for k in id2doc}
# %%
def get_envs():
    from dotenv.main import dotenv_values
    envs = dotenv_values('../.env')
    return envs
envs = get_envs()
# %%
tk_rba = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space = True)
tk_bt = AutoTokenizer.from_pretrained('bert-base-uncased')
tk_lama = AutoTokenizer.from_pretrained(envs['LLAMA2_7B'])
tk_gpt = AutoTokenizer.from_pretrained('gpt2')
# %%
qa_data = json.load(open(Path(envs['CUAD_PATH']) / 'CUAD_v1.json'))['data']
# %%
examples = create_examples(qa_data)
# %%
exa_path = '../data/examples.pkl'
if True or not Path(exa_path).exists():
    with open(exa_path, 'wb') as f:
        pickle.dump(examples, f)
# %%
examples = pickle.load(open(exa_path, 'rb'))
# %%
def load_docs(path):
    print('read')
    doc_objs = pickle.load(open(path, 'rb'))
    print('build')
    return [DocTokens(**k, doc_text = id2doc[k['doc_id']]) for k in doc_objs]
# %%
doc_objs_rba = load_docs('../data/doc/doc_tokens_roberta_rm.pkl')
# %%
feats = convert_features(examples[2], doc_objs_rba[0],
                         max_seq_length= 512,
                 doc_stride = 256, max_query_length=128, is_training=True, 
                 tokenizer=tk_bt)
# %%
print(examples[2]['answer_texts'][0])
feat = [f for f in feats if not f['is_impossible']][0]
st_pos = feat['start_position']
ed_pos = feat['end_position']
aw = tk_bt.decode(feat['input_ids'][st_pos: ed_pos + 1])
print(aw)
# %%
chr_st, chr_ed = examples[2]['answer_spans'][0]
char_to_token = convert_token_char_map(doc_objs_rba[0]['token_to_char'], len(id2doc[0]['doc']))
answer_st = char_to_token[chr_st][0]
answer_end = char_to_token[chr_ed-1][1] - 1
# %%
print(chr_st, chr_ed)
print(answer_st, answer_end)
print(st_pos, ed_pos)
# %%
token_to_char = doc_objs_rba[0]['token_to_char']
doc_tids = doc_objs_rba[0]['doc_token_ids']
doc = id2doc[0]['doc']
# %%
# Test process_features
feats = process_features(examples[:10], doc_objs_rba, tk_lama, 
                        max_seq_length= 512,
                        doc_stride = 256, max_query_length=128,)
# %%
# feat = [f for f in feats if not f['is_impossible']][0]
for feat in feats:
    if feat['is_impossible']:
        continue
    st_pos = feat['start_position']
    ed_pos = feat['end_position']
    aw = tk_lama.decode(feat['input_ids'][st_pos: ed_pos + 1])
    ori_text = examples[feat['example_index']]['answer_texts'][0]
    if ori_text != aw:
        print(f'Example: {feat["example_index"]} {ori_text} -> {aw}')
# %%
# check failed spans
feats_rba = pickle.load(open('../data/qa_features/qa_roberta.pkl', 'rb'))
# %%
exa2feat = [[] for _ in range(len(examples))]
for feat in feats_rba:
    exa2feat[feat['example_index']].append(feat)

# %%
fail_e_index = []
for i, exa in enumerate(examples):
    if not exa['is_impossible'] and len(exa2feat[i]) == 0:
        fail_e_index.append(i)
# %%
exa = examples[388]
doc_o = [k for k in doc_objs_rba if k['doc_id'] == exa['doc_id']][0]
chr_st, chr_ed = exa['answer_spans'][0]
char_to_token = doc_o['char_to_token']
answer_st = char_to_token[chr_st][0]
answer_end = char_to_token[chr_ed-1][1] - 1
# %%
