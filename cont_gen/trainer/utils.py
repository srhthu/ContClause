"""Utilities for training"""

import torch
from collections import defaultdict
from typing import Dict, Union

def group_nodecay_parameters(model, weight_decay = 0.0, no_decay = ['bias', 'LayerNorm.weight']):
    """
    Return parameter groups of decay parameters and no_decay parameters
    """
    named_params = list(model.named_parameters())
    nd_param_names = set([n for n, _ in named_params if any(nd in n for nd in no_decay)])
    param_groups = [
        {
            'params': [p for n,p in named_params if n not in nd_param_names],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n,p in named_params if n in nd_param_names],
            'weight_decay': 0.0
        }
    ]
    return param_groups

def get_smart_optimizer(model, lr, weight_decay = 0.0, **kws):
    param_groups = group_nodecay_parameters(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr = lr, **kws)
    return optimizer

class AccumulateTensors:
    """For average variables across log setps """
    def __init__(self):
        self._values = defaultdict(float)
        self._count = 0
    
    def add(self, tensor_dt):
        for name, tensor in tensor_dt.items():
            if isinstance(tensor, torch.Tensor):
                value = tensor.detach().cpu()
                value = value.item() if value.reshape(-1).shape[0] == 1 else None
            else:
                value = None
            if value is not None:
                self._values[name] += value
        self._count += 1
    def get(self):
        return {k:v / self._count for k,v in self._values.items()}

class AverageTensors:
    """
    To hold step outputs, only keep scales and output their average.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.records = None
        self.step_count = 0

    def filter_tensor(self, x):
        """keep tensors with only one element"""
        return (isinstance(x, torch.Tensor) and x.numel() == 1)

    def record(self, tensor_dt):
        # do not merge
        new_dt = {k:v.detach().cpu().squeeze() for k,v in tensor_dt.items() if self.filter_tensor(v)}
        if self.records is None:
            self.records = new_dt
        else:
            for k,v in new_dt.items():
                self.records[k] += v
        self.step_count += 1
    
    def average(self)->Dict[str, float]:
        """Return average and reset history records"""
        ave_ts = {k:(v / self.step_count).item() for k,v in self.records.items()}
        self.reset()
        return ave_ts

def number2str(x: Union[int, float]):
    if abs(x) > 0.001 and abs(x) < 1000:
        return f'{x:.5g}'
    elif abs(x) < 1e-6:
        return '0.0'
    else:
        return f'{x:.3e}'