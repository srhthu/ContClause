"""
Utilities for model, tokenizer, etc.
"""
import argparse
import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import time
from transformers import (
    AutoTokenizer, AutoConfig, PreTrainedModel,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
    Trainer, GPT2LMHeadModel, LlamaForCausalLM, T5Model,
    BertLMHeadModel, BartForCausalLM
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
import deepspeed
from deepspeed.utils import safe_get_full_fp32_param
import accelerate
from accelerate import PartialState, Accelerator
from accelerate.utils import DummyOptim
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Union, Tuple



def load_and_update_ds_config(ds_dict_or_path, device_bs, grad_acc_steps):
    """Load deepspeed config and config batch size fields"""
    if ds_dict_or_path is None:
        return None
    # load json deepspeed config file
    ds_config = ds_dict_or_path if isinstance(ds_dict_or_path, dict) \
                else json.load(open(ds_dict_or_path))
    
    # update bath_size fields
    ds_config['train_micro_batch_size_per_gpu'] = device_bs
    ds_config['gradient_accumulation_steps'] = grad_acc_steps
    if 'train_batch_size' in ds_config:
        # keep two variables and remove the left one
        _ = ds_config.pop('train_batch_size')
    return ds_config

def get_hf_ds_plugin(ds_config):
    assert ds_config is not None
    from transformers.deepspeed import HfTrainerDeepSpeedConfig

    hf_deepspeed_config = HfTrainerDeepSpeedConfig(ds_config)

    # Accelerate DeepSpeed Plugin
    from accelerate.utils import DeepSpeedPlugin

    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=hf_deepspeed_config)
    return deepspeed_plugin


def initialize_accelerator(
    ds_config, device_bs, grad_acc_steps
)->Tuple[Accelerator, dict]:
    """
    Load deepspeed config and initialize accelerator
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
            deepspeed_plugin=deepspeed_plugin)
    return accelerator, ds_config

def build_model(
    name_or_path, 
    torch_dtype, 
    local_rank, 
    is_zero3 = False
)->PreTrainedModel:
    """
    Build and initialize model, handle device_map.

    Args:
        is_zero3: if use deepspeed zero3, device_map should not be specified.
    """
    config = AutoConfig.from_pretrained(name_or_path, trust_remote_code = True)
    kws = dict(
        torch_dtype = torch_dtype,
        device_map = local_rank,
        trust_remote_code = True
    )
    if is_zero3:
        _ = kws.pop('device_map')

    if 'phi' in name_or_path:
        # use flash attention for microsoft/phi-1_5
        kws.update(dict(flash_attn=True, flash_rotary=True, fused_dense=True))
    
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, **kws)
    else:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kws)
    return model

def build_lora_model_demo(model):
    """
    Build lora model given the base model. This is used for demo.
    """
    task_type = "SEQ_2_SEQ_LM" if model.config.is_encoder_decoder else "CAUSAL_LM"
    config = LoraConfig(
        r=8,
        lora_alpha= 16 ,
        target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model.config.model_type],
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, config)