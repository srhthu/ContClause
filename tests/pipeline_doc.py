"""
Test the pipeline of document processing.
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

from cont_gen.data_process import cut_doc
# %%
# reload the module if changed
reload(cut_doc)
from cont_gen.data_process.cut_doc import (
    remove_space_keep_mapping,
    doc_tokenization_fast, 
    convert_all_documents,
    process_cuad_data
)
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
# Test the funcion of remove_space_keep_mapping
s = '123     456\n\n78 panda 滴 µ'
def check_map(s):
    new_s, c2o_span = remove_space_keep_mapping(s)
    for i,span in enumerate(c2o_span):
        assert s[span[0]] == new_s[i]
print(remove_space_keep_mapping(s))
check_map(s)
# %%
# Load the dataset
qa_data = json.load(open(Path(envs['CUAD_PATH']) / 'CUAD_v1.json'))['data']
doc_ids = [e['title'] for e in qa_data]
docs = [e['paragraphs'][0]['context'] for e in qa_data]
# %%
# Test the function of doc_tokenization_fast
# Print tokens and original spans
tk = tk_lama
doc_tids, token_to_char = doc_tokenization_fast(s, tk, remove_space = False)
for tid, span in zip(doc_tids, token_to_char):
    token = tk.convert_ids_to_tokens([tid])[0]
    print(f'{repr(token)}->{repr(s[span[0]: span[1]])}')
# %%
# Test the multiprocessing func: process_cuad_data
doc_map = process_cuad_data(qa_data[:100], tk, True, 10)
# %%
# Save the doc_id and doc information
doc_text = [{'doc_id': e['title'], 'doc': e['paragraphs'][0]['context']} for e in qa_data]
d_id_path = Path('../data/doc/doc_id_text.pkl')
if not d_id_path.exists():
    with open(d_id_path, 'wb') as f:
        pickle.dump(doc_text, f)
# %%
