"""Utilities for distributed training"""
import json
import os
import logging
from pathlib import Path
import time
from typing import Tuple
from transformers.deepspeed import HfTrainerDeepSpeedConfig
from accelerate import PartialState, Accelerator
from accelerate.utils import DeepSpeedPlugin

def is_state_initialized()->bool:
    """Whether accelerate State is initialized"""
    return len(PartialState._shared_state) > 0

def load_and_update_ds_config(ds_dict_or_path, device_bs, grad_acc_steps):
    """
    Load deepspeed config and modify the config dict.

    Ignore the field defined below.
    """
    ignore_fields = [
        'train_micro_batch_size_per_gpu', 
        'train_batch_size', 
        'gradient_accumulation_steps', 
        'optimizer', 
        'scheduler'
    ]
    if ds_dict_or_path is None:
        return None
    # load json deepspeed config file
    ds_config = ds_dict_or_path if isinstance(ds_dict_or_path, dict) \
                else json.load(open(ds_dict_or_path))
    
    # ignore fields
    ignored_key_values = {k:ds_config[k] for k in ignore_fields if k in ds_config}
    print(f'The following fields in ds_config will be ignored: {ignored_key_values}')
    ds_config = {k:v for k,v in ds_config.items() if k not in ignore_fields}

    # config ignored fields related to batch size
    ds_config['train_micro_batch_size_per_gpu'] = device_bs
    ds_config['gradient_accumulation_steps'] = grad_acc_steps

    return ds_config

def get_hf_ds_plugin(ds_config):
    """Convert ds_config dict to huggingface deepspeed plugin"""
    # Code is adapted from transformers.trainer.Trainer
    assert ds_config is not None

    hf_deepspeed_config = HfTrainerDeepSpeedConfig(ds_config)

    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=hf_deepspeed_config)
    return deepspeed_plugin

def initialize_accelerator(
    ds_config, device_bs = None, grad_acc_steps = None
)->Tuple[Accelerator, dict]:
    """
    Load deepspeed config and initialize accelerator.

    The optimizer and scheduler fields in ds_config will be ignored.

    The batch size related parameters will be re-configured.
    """
    ds_config = load_and_update_ds_config(
        ds_config, device_bs, grad_acc_steps
    )
    if ds_config is None:
        accelerator = Accelerator(gradient_accumulation_steps=grad_acc_steps)
    else:
        if ds_config.get('zero_optimization',{}).get('stage') == 3:
            # solve the error with accelerate
            ds_config['train_batch_size'] = 'auto'
        deepspeed_plugin = get_hf_ds_plugin(ds_config)
        accelerator = Accelerator(
            gradient_accumulation_steps=grad_acc_steps,
            deepspeed_plugin=deepspeed_plugin
        )
    return accelerator, ds_config

class DistLogger:
    """Logger for distributed training. Compatible with tqdm bar"""
    def __init__(self, file = None):
        self.file = file
        assert is_state_initialized(), 'Should initialize PartialState before.'
        self.state = PartialState()
        if file is not None:
            Path(file).resolve().parent.mkdir(exist_ok = True, parents=True)
        self.tqdm_bar = None
    
    def log(self, msg, main_only = True):
        if main_only and self.state.is_main_process:
            # log at main process
            self._log(msg, '[Main]')
        elif not main_only :
            # log at every process
            self._log(msg, f'[Process {self.state.process_index}]')
    
    def _log(self, msg, proc_field):
        msg = self.decorate(msg, proc_field)
        # log to stdout
        if self.tqdm_bar is None:
            print(msg)
        else:
            self.tqdm_bar.write(msg)
        # log to file
        if self.file is not None:
            with open(self.file, 'a') as f:
                f.write(msg + '\n')
    
    def decorate(self, msg, proc_field):
        """Add timestamp, etc. to message"""
        return f'[{self.get_timestamp()}]{proc_field} {msg}'

    def get_timestamp(self):
        time_s = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        return time_s

    def add_tqdm(self, bar):
        self.tqdm_bar = bar
    
    def remove_tqdm(self):
        self.tqdm_bar = None