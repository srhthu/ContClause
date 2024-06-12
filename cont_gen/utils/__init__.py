# from .io import (
#     load_jsonl, load_json, save_jsonl,
#     load_pickle, save_pickle
# )
from pathlib import Path
import pandas as pd
import re
from .io import *
from .plot_loss import plot_multiple_loss

def get_ckpt_paths(run_dir) -> List[Path]:
    """Return the complete path of sub dirs of run_dir"""
    run_dir = Path(run_dir)
    ckpt_dirs = list(run_dir.glob('checkpoint-*'))
    # sort by global step
    def get_step(name):
        step = int(re.search(r'checkpoint-(.*)', name).group(1))
        return step
    step_path = [(get_step(k.name), k) for k in ckpt_dirs]
    step_path.sort(key = lambda k: k[0])

    return [k[1] for k in step_path]

def get_loss_df_norm(run_dir)->pd.DataFrame:
    """Return setp and loss where step is normalized by steps_pre_epoch"""
    first = get_ckpt_paths(run_dir)[0]
    steps_per_epoch = int(Path(first).name.split('-')[1])
    loss_df = get_loss_df(run_dir)
    loss_df['step'] = loss_df['step'] / steps_per_epoch
    return loss_df

def get_loss_df(run_dir)->pd.DataFrame:
    """
    Return a dataframe with fields of step and loss
    """
    # get loss from loss json file or log file
    loss_fn = Path(run_dir) / 'loss_log.jsonl'
    if loss_fn.exists():
        steps, losses = load_loss_json(loss_fn)
        return pd.DataFrame({'step': steps, 'loss': losses})
    
    log_fn = Path(run_dir) / 'log.txt'
    if log_fn.exists():
        lines = open(log_fn).readlines()

        loss_logs = []
        for line in lines:
            if line.startswith('{'):
                try:
                    loss_logs.append(eval(line))
                except:
                    print(line)
        x = [eval(k['step'].strip("'")) for k in loss_logs]
        y = [eval(k['loss'].strip("'")) for k in loss_logs]
        return pd.DataFrame({'step': x, 'loss': y})

def load_loss_json(path):
    logs = load_jsonl(path)
    steps = [k['step'] for k in logs]
    losses = [k['loss'] for k in logs]

    # incase the scripts run many times, keep the last run
    # from end to front to find position with asend step
    start_idx = len(steps) - 1
    while start_idx > 0:
        if steps[start_idx] > steps[start_idx - 1]:
            start_idx -= 1
        else:
            break
    return steps[start_idx:], losses[start_idx:]