"""Utilities for training"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, Union, Mapping
from accelerate import PartialState

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

def nested_to_cpu(tensors):
    "Transfer `tensors` to cpu (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_to_cpu(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_to_cpu(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t

def convert_column_to_row(batch_results):
    """Convert data grouped by column to row samples."""
    if isinstance(batch_results, Mapping):
        keys = list(batch_results.keys())
        sample_results = list(zip(*batch_results.values()))
        return [dict(zip(keys, r)) for r in sample_results]

    elif isinstance(batch_results, (list, tuple)):
        return list(zip(*batch_results))

def compute_clm_loss_with_ignore(model, batch, ignore_index):
    # out = model(**batch)
    out = model(input_ids = batch['input_ids'])
    logits = out['logits']
    labels = batch['input_ids']
    
    # Copy from Llama
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    return loss