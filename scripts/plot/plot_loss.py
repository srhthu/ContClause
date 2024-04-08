"""Plot the training loss"""
# %%
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# %%
def get_loss(path):
    lines = open(path).readlines()

    loss_logs = []
    for line in lines:
        if line.startswith('{'):
            try:
                loss_logs.append(eval(line))
            except:
                print(line)
    x = [eval(k['step'].strip("'")) for k in loss_logs]
    y = [eval(k['loss'].strip("'")) for k in loss_logs]
    return x, y
def load_jsonl(path):
    with open(path) as f:
        data = [json.loads(k) for k in f]
    return data

def load_loss_json(path):
    logs = load_jsonl(path)
    steps = np.array([k['step'] for k in logs])
    losses = np.array([k['loss'] for k in logs])

    # incase the scripts run many times, keep the last run
    # from end to front to find position with asend step
    start_idx = len(steps) - 1
    while start_idx > 0:
        if steps[start_idx] > steps[start_idx - 1]:
            start_idx -= 1
        else:
            break
    return steps[start_idx:], losses[start_idx:]
# %%
x_1, y_1 = load_loss_json('../../runs/genqa/flan-t5-large_quest_lr1e-4_bs16/loss_log.jsonl')
x_2, y_2 = load_loss_json('../../runs/genqa/flan-t5-large_quest_lr1e-4_bs16_reformat/loss_log.jsonl')

x_3, y_3 = load_loss_json('../../runs/genqa/flan-t5-large_quest_lr1e-4_bs16_re1/loss_log.jsonl')
x_4, y_4 = load_loss_json('../../runs/genqa/flan-t5-large_quest_lr1e-4_bs16_fix/loss_log.jsonl')
x_5, y_5 = load_loss_json('../../runs/genqa/flan-t5-large_quest_lr1e-4_bs16_new/loss_log.jsonl')
x_6, y_6 = load_loss_json('../../runs/genqa/flan-t5-large_quest_lr1e-4_bs16_reformat_tgt512/loss_log.jsonl')
# %%
plt.plot(x_1 / 2, y_1, label = 'prev')
# plt.plot(x_3 / 2, y_3, label = 're1')
plt.plot(x_2, y_2, label = 'reformat')
plt.plot(x_4, y_4, label = 'fix')
# plt.plot(x_5 / 2, y_5, label = 'new')
plt.plot(x_6 / 2, y_6, label = 'reformat_tgt512')
plt.ylim((-0.05, 0.6))
plt.legend()
# %%
# plot loss_log
logs = load_jsonl('../../runs/mem_genqa/flan-t5-large_quest_lr1e-4_bs16/loss_log.jsonl')
# %%
plt.plot(*list(zip(*[(k['step'], k['loss']) for k in logs])))
# %%
t = list(zip([(k['step'], k['loss']) for k in logs]))
# %%
