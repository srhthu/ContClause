"""Test the cuad dataset with status"""
# %%
import sys
sys.path.insert(0, '../')
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path

from clm.data_loader import CUAD_FileLoader_Clean, get_sorted_child
# %%
tk = AutoTokenizer.from_pretrained('microsoft/phi-1_5', trust_remote_code = True)
tk.add_special_tokens({'pad_token': '[PAD]'})
# %%
ds = CUAD_FileLoader_Clean('../../data/cuad_contracts_small', tk, 1024, 42)
# %%
log_step = 1000
max_step = log_step * 5
bar = tqdm(total = len(ds.all_files), ncols = 80)
log_history = []
for i, d in enumerate(ds):
    bar.update(ds.status['file_index']+1 - bar.n)
    # print(ds.status)
    # break
    if (i+1) % log_step == 0:
        logs = {**ds.status}
        logs['chunk_per_doc'] = (logs['index'] + 1) / (logs['file_index'] + 1)
        log_history.append(logs)
        bar.write(str(logs))
    if (i+1) >= max_step:
        break
bar.close()
# %%
get_sorted_child('../../data/cuad_contracts', 'file')
# %%
ds.set_status(log_history[-1])
print(ds.status)
# %%
ds.reset()
# %%
chunks = list(ds.process_doc(open(Path('../../data/cuad_contracts') / '2005/000009588.txt').read()))
print(len(chunks))
# %%
all_data = []
for d in ds:
    all_data.append({**ds.status})
# %%
all_data[0].keys()
# %%
pos2id = {(d['file_index'], d['chunk_index']): i for i, d in enumerate(all_data)}
# %%
pos2id[(3, 151)]
# %%
ds.set_status({'index': 163, 'file_index': 3, 'file_name': '2010/000035965.txt', 'chunk_index': 151})
left_data = [{**ds.status} for d in ds]
# %%
left_data[0]
# %%
pos2id[(3, 151)]
# %%
