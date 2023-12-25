# %%
import context
from pathlib import Path
import re
from tqdm import tqdm
from multiprocessing import Pool
# %%
data_dir = Path('../../data/cuad_contracts')
# %%
def estimate_training_cost(seq_len, micro_bs, step_time):
    total_byte = 35 * 1024**3
    total_token = total_byte / 10
    total_micro_steps = total_token / seq_len / micro_bs
    total_time = total_micro_steps * step_time / 3600
    return total_micro_steps, total_time
# %%
estimate_training_cost(1024, 4 * 4, 1.247)
# %%
def get_file_size(name):
    return len(open(name).read())
# %%
counts = []
file_size = {} # num of bytes
for sub_dir in data_dir.glob('*'):
    print(f'Reading {sub_dir.name}')
    files = list(sub_dir.glob('*'))
    file_num = len(files)
    print(f'Total {file_num} files')
    counts.append((sub_dir.name, file_num))
    with Pool(10) as p:
        for fn, sz in zip(files, 
                          p.imap(get_file_size, tqdm(files, desc = sub_dir.name), chunksize = 100)):
            file_size[fn] = sz
    # for fname in tqdm(files, desc = sub_dir.name):
        # file_size[str(fname)] = len(open(fname).read())
# %%
tot_byte = sum(file_size.values())
# %%
tot_mb = tot_byte / 1024 **2
print(tot_mb)
# %%
len(file_size)
# %%
tot_byte / len(file_size)
# %%
n_token_perlog = 1024 * 4 * 1 * 20
# %%
n_log = tot_byte / 10 / n_token_perlog
print(n_log)
dur = 8
est_time = n_log * dur / 3600
print(f'estimate training hour: {est_time}')
# %%
3600 / 8 * 80
# %%
'80k -> 36 M / hour -> 100 hour'