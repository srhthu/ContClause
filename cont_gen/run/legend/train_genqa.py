"""
Train generative QA
"""
import argparse
import sys
import re
import shutil
from pathlib import Path
import json
import torch
import numpy as np
import time
from functools import partial
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
from accelerate.utils import DummyOptim, DistributedType
import torch.nn as nn
from torch.utils.data import DataLoader
from pynvml import *
import torch.distributed as dist

from cont_gen.trainer.utils_dist import initialize_accelerator, DistLogger
from cont_gen.data_loader.cuad_prompt import CUAD_SFT, SFT_Padding, CUAD_SFT_Seq2Seq
from cont_gen.utils.model_utils import build_hf_or_peft_model, smart_resize_embeddings
from cont_gen.trainer.utils import get_smart_optimizer, compute_clm_loss_with_ignore
from cont_gen.trainer.train_only_accelerate import Trainer_Basic, TrainingArgs_Basic

TORCH_DTYPE_MAP = {
    'fp16': torch.float16, 
    'bf16': torch.bfloat16, 
    'fp32': torch.float32
}

class LM_Loss_With_Ignore:
    def __init__(self, ignore_index):
        self.ignore_index = ignore_index
    
    def __call__(self, model, batch):
        loss = compute_clm_loss_with_ignore(model, batch, self.ignore_index)
        return {'loss': loss}

class LM_Feed:
    def __call__(self, model, batch):
        batch = {k:v for k,v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        loss = model(**batch).loss
        return {'loss': loss}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--ds_config')
    parser.add_argument('--debug', action = 'store_true')
    # data args
    parser.add_argument('--data_path', help = 'prompt file path')
    parser.add_argument('--max_length', type = int, default = 1200)
    # model args
    parser.add_argument('--base_model')
    parser.add_argument('--saved_model', help = 'path of saved model state dict',
                        nargs='?', const = None, default = None)
    parser.add_argument('--dtype', choices = ['fp16', 'bf16', 'fp32'], default = 'bf16')
    parser.add_argument('--labels_on_full', action = 'store_true')
    parser.add_argument('--lora', action = 'store_true', help = 'use lora adapter')
    parser.add_argument('--lora_r', type = int, default=8)
    parser.add_argument('--lora_alpha', type = int, default=16)
    parser.add_argument('--lora_target_modules', nargs='+', default = None)
    parser.add_argument('--lora_dropout', type = float, default = 0.05)
    # optimizer
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--weight_decay', type = float, default = 0.0)
    # training args
    parser.add_argument('--device_batch_size', type = int)
    parser.add_argument('--grad_acc_steps', type = int)
    parser.add_argument('--max_epochs', type = int)
    parser.add_argument('--max_steps', type = int)
    parser.add_argument('--logging_steps', type = int, default = 5, help = 'macro steps')
    parser.add_argument('--save_steps', type = int, help = 'micro steps')
    parser.add_argument('--save_epochs', type = int, default = 1)
    parser.add_argument('--save_total_limit', type = int, default = 5)
    
    # to be compatible with deepspeed launch
    parser.add_argument('--local_rank')
    return parser
    

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Initialize distribution environment
    accelerator, ds_config = initialize_accelerator(
        args.ds_config, args.device_batch_size, args.grad_acc_steps
    )

    # training args
    tr_args = TrainingArgs_Basic(
        device_batch_size = args.device_batch_size,
        grad_acc_steps=args.grad_acc_steps,
        max_epochs = args.max_epochs,
        max_steps = args.max_steps,
        logging_steps = args.logging_steps,
        save_steps = args.save_steps,
        save_epochs = args.save_epochs,
        save_total_limit = args.save_total_limit
    )
    args.total_batch_size = tr_args.total_batch_size
    
    # Get logger
    log_file = None if args.output_dir is None else str(Path(args.output_dir) / 'log.txt')
    logger = DistLogger(file = log_file)

    # log arguments
    args_str = json.dumps(args.__dict__, indent = 4, ensure_ascii=False)
    if args.output_dir:
        with open(Path(args.output_dir) / 'args.json', 'w') as f:
            f.write(args_str)
    logger.log(f'Training args:\n {args_str}')

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code = True)
    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load dataset
    ds_cls = CUAD_SFT if 't5' not in args.base_model else CUAD_SFT_Seq2Seq
    dataset = ds_cls(
        args.data_path, tokenizer = tokenizer, 
        max_length = args.max_length, 
        labels_on_full = args.labels_on_full,
        small = args.debug
    )
    logger.log(f'dataset: {dataset.__class__.__name__}')
    if args.debug:
        # pad to max_length to test memory
        collate_fn = SFT_Padding(tokenizer.pad_token_id, 
                                 pad_side = 'right', 
                                 pad_to_max_len = args.max_length)
    else:
        # dynamic padding
        collate_fn = SFT_Padding(tokenizer.pad_token_id, pad_side = 'right')

    # Build model
    if args.lora:
        peft_config = LoraConfig(
            r = args.lora_r, lora_alpha = args.lora_alpha,
            target_modules = args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias = "none"
        )
        # task_type will be configured later
    else:
        peft_config = None
    model = build_hf_or_peft_model(
        args.base_model, accelerator, TORCH_DTYPE_MAP[args.dtype], 
        peft_config=peft_config
    )
    # load saved state dict
    if args.saved_model:
        logger.log(f'Load saved model parameters from: {args.saved_model}')
        with open(args.saved_model, 'rb') as f:
            state_dict = torch.load(f, map_location = accelerator.device)
        model.load_state_dict(state_dict)
        del state_dict

    # resize model embedding if necessary
    smart_resize_embeddings(tokenizer, model, logger)
    accelerator.wait_for_everyone()
    # logger.log(f'{model.get_input_embeddings().weight.shape}', main_only = False)
    # exit()
    # batch = collate_fn([dataset[i] for i in range(5)])
    # logger.log(str(batch['input_ids'].max()), main_only = False)
    # print(model.transformer.wte(batch['input_ids'].cuda()).shape)
    # exit()

    # build optimizer
    optimizer = get_smart_optimizer(model, args.lr, args.weight_decay)

    # Trainer
    trainer = Trainer_Basic(
        tr_args, model, dataset, optimizer, accelerator,
        ds_config = ds_config,
        collate_fn = collate_fn,
        compute_loss_fn = LM_Feed(),
        output_dir = args.output_dir,
        logger = logger
    )

    trainer.train()

if __name__ == '__main__':
    main()