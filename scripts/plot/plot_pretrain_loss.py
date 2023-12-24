"""Plot the training loss under runs/pretrain_debug"""
# %%
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# %%
def get_loss(path, line_n = -1):
    lines = open(path).readlines()
    logs = json.loads(lines[line_n])
    x = [k['step'] for k in logs]
    y = [k['loss'] for k in logs]
    return x, y
# %%
run_dir = Path('../../runs/pretrain_debug')
# %%
# each setting is a tuple of (device, device_bs, grad_acc_steps)
settings = [
    (1,8,2), (2,4,2), (2,8,2), (2,16,2),
    (2,32,2), (4,2,2)
]
run_dir = Path('../../runs/pretrain_debug')
results = {k:get_loss(run_dir / 'device{}_devbs{}_gaccstep{}.log'.format(*k)) for k in settings}
# %%
results = {}
for lr in ['1e-4', '1e-5']:
    log_name = f'phi-1_5_len1024_lr{lr}_gpu4_devbs1_gaccstep_4.log'
    results[lr] = get_loss(run_dir / log_name)
results['clean_1e-5'] = get_loss(run_dir / 'clean_phi-1_5_len1024_lr1e-5_gpu4_devbs1_gaccstep_4.log')
# %%
plt.plot(*results[(1,8,2)], label = '1,8,2')
plt.plot(*results[(2,4,2)], label = '2,4,2')
plt.plot(*results[(4,2,2)], label = '4,2,2')
plt.legend()
# %%
# only change device_batch_size
for dbs in [4,8,16,32]:
    plt.plot(*results[(2,dbs,2)], label = f'2,{dbs},2')
plt.legend()
# %%
# plot all results
plt.figure(figsize = (15,5))
for k,v in results.items():
    plt.plot(*v, label = str(k))
plt.legend()
# %%
