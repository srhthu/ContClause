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
# %%
x_1, y_1 = get_loss('../../runs/qa/roberta-base_lr1e-4_bs16/logs.txt')
x_2, y_2 = get_loss('../../runs/qa/roberta-base_lr1e-5_bs16/logs.txt')
# %%
plt.plot(x_1, y_1)
plt.plot(x_2, y_2)
# %%
