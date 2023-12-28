"""Test the cuad dataset with status"""
# %%
import sys
sys.path.insert(0, '../')
from transformers import AutoTokenizer
from tqdm import tqdm

from clm.data_loader import CUAD_FileLoader_Clean, get_sorted_child
# %%
tk = AutoTokenizer.from_pretrained('microsoft/phi-1_5', trust_remote_code = True)
tk.add_special_tokens({'pad_token': '[PAD]'})
# %%
ds = CUAD_FileLoader_Clean('../../data/cuad_contracts', tk, 1024, 43)
# %%
log_step = 1000
max_step = log_step * 20
bar = tqdm(total = len(ds.all_files), ncols = 80)
log_history = []
for i, d in enumerate(ds):
    bar.update(ds.status['file_index'] - bar.n)
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
ds.set_status(log_history[0])
print(ds.status)
# %%
