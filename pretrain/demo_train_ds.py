"""
A demo of pretraining using deepspeed.

Aim:
    providing a toy example
    investigating memory usage of different input seq length.
"""
import argparse
import sys
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
from pynvml import *

from clm.data_loader import FileLoader, FileLoader_Clean
from clm.utils import (
    get_gpu_utilization, print_gpu_utilization, logger,
    DistLogger, smart_resize_embeddings, ParamChangeChecker
)
from clm.prepare import (
    load_and_update_ds_config, get_hf_ds_plugin,
    build_model, build_lora_model_demo, initialize_accelerator
)

def compute_loss(model, batch, ignore_index):
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


def train_with_deepspeed(args, model, dataloader, ds_config, ignore_index):
    logger.log_main('Build deepspeed engine')
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model = model,
        model_parameters = model.parameters(),
        config = ds_config
    )
    training_loop_deepspeed(args, model_engine, dataloader, ignore_index)

def training_loop_deepspeed(args, model, dataloader, ignore_index):
    """
    Model is a deepspeed engine.
    """
    data_iter = iter(dataloader)
    loss_host = 0.0
    model.train()

    checker = ParamChangeChecker(model, use_deepspeed=True)
    checker.check()

    logs = []
    last_log_time = time.time()

    for step in range(args.max_steps):
        batch = next(data_iter)
        loss = compute_loss(model, batch, ignore_index = ignore_index)
        loss_host += loss.detach().cpu().item()
        # Modified. Replace loss.backward()
        model.backward(loss)
        model.step()

        # main_log((
        #     ("Update" if checker.check() else "No update")
        #     + f' at step {step}'
        #     + f' is acc boundary: {model.is_gradient_accumulation_boundary()}'
        # ))

        if (step+1) % (args.grad_acc_steps * args.log_steps) == 0:
            # handle logging
            dur = time.time() - last_log_time
            last_log_time = time.time()

            logger.log_main(f'Step {step+1} loss: {loss_host:.4f} time: {dur:.2f}')
            logs.append({'step': step, 'loss': loss_host, 'time': dur})
            loss_host = 0.0
    return logs

def train_ddp_accelerate(args, accelerator: Accelerator, model, dataloader, ignore_index):
    """
    Train with DDP.
    """
    data_iter = iter(dataloader)
    loss_host = 0.0

    # Build optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    
    # prepare model and optimizer
    # enable checkpointing
    # model.gradient_checkpointing_enable()

    model, optimizer = accelerator.prepare(model, optimizer)
    logger.log_main('Finish prepare')
    print_gpu_utilization()
    
    logger.log_main(f'{type(model)}')

    model.train()

    logs = []
    last_log_time = time.time()

    checker = ParamChangeChecker(
        model, 
        use_deepspeed=accelerator.state.deepspeed_plugin is not None
    )
    # checker.check()
    # exit()

    for step in range(args.max_steps):
        batch = next(data_iter)
        # print(f'input shape: {batch["input_ids"].shape}', flush = True)
        # exit()
        with accelerator.accumulate(model):
            loss = compute_loss(model, batch, ignore_index = ignore_index)
        loss_host += loss.detach().cpu().item()
        
        # backward
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # main_log((
        #     ("Update" if checker.check() else "No update")
        #     + f' at step {step}'
        # ))

        if (step+1) % (args.grad_acc_steps * args.log_steps) == 0:
            loss_host /= (args.log_steps * args.grad_acc_steps)
            dur = time.time() - last_log_time
            last_log_time = time.time()

            logger.log_main(f'Step {step+1} loss: {loss_host:.4f} time: {dur:.2f}')
            logs.append({'step': step, 'loss': loss_host, 'time': dur})
            loss_host = 0.0
    
    return logs, model

def collate_fn(samples):
    # For each key, gather the values of samples
    batched = dict(zip(samples[0].keys(),
                        zip(*[[sample[k] for k in sample] for sample in samples])))
    batched = {k: torch.tensor(v) for k,v in batched.items()}
    return batched

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--dtype', choices = ['fp16', 'bf16', 'fp32'], default = 'fp16')
    parser.add_argument('--ds_config')
    parser.add_argument('--lora', action = 'store_true')
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--max_steps', type = int, default = 30)
    parser.add_argument('--log_steps', type = int, default = 5)
    parser.add_argument('--device_bs', type = int, default = 2)
    parser.add_argument('--grad_acc_steps', type = int, default = 2)
    parser.add_argument('--max_length', type = int, default = 512)

    parser.add_argument('--lib', choices = ['deepspeed', 'accelerate'], help = 'training framework to use')
    # to be compatible with deepspeed launch
    parser.add_argument('--local_rank')
    args = parser.parse_args()

    torch_dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

    accelerator, ds_config = initialize_accelerator(
        args.ds_config, args.device_bs, args.grad_acc_steps
    )

    state = PartialState()

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code = True)
    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load dataset
    ds_cls = FileLoader_Clean
    dataset = ds_cls(
        './data/cuad_contracts', 
        tokenizer = tokenizer, max_length = args.max_length
    )
    
    dataloader = DataLoader(dataset, batch_size = args.device_bs, collate_fn = collate_fn)
    dataloader = accelerator.prepare(dataloader)

    # Build model
    is_zero3 = False if accelerator.state.deepspeed_plugin is None else \
            accelerator.state.deepspeed_plugin.zero3_init_flag
    model = build_model(args.model_path, torch_dtype_map[args.dtype], 
                        state.local_process_index, 
                        is_zero3 = is_zero3)
    # model = build_model(args.model_path, torch.float32, state.local_process_index)

    # smart_resize_embeddings(tokenizer, model)

    # Lora
    if args.lora:
        model = build_lora_model_demo(model)

    print_gpu_utilization()
    time.sleep(3)

    if args.lib == 'deepspeed':
        train_with_deepspeed(args, model, dataloader, ds_config, ignore_index=tokenizer.pad_token_id)
    elif args.lib == 'accelerate':
        logs, engine = train_ddp_accelerate(args, accelerator, model, dataloader, ignore_index = tokenizer.pad_token_id)

        all_time = [k['time'] for k in logs[1:]]
        logger.log_main(f'Average log duration: {np.mean(all_time)}')

        # out_dir = 'runs/pretrain_debug'
        # if accelerator.is_local_main_process:
        #     Path(out_dir).mkdir(parents = True, exist_ok = True)
        #     # log_file = f'device{accelerator.num_processes}_devbs{args.device_bs}_gaccstep{args.grad_acc_steps}.log'
        #     log_file = 'clean_phi-1_5_len1024_lr1e-5_gpu4_devbs1_gaccstep_4.log'
        #     with open(Path(out_dir) / log_file, 'a') as f:
        #         f.write(json.dumps(logs) + '\n')
            
        
        # engine.save_checkpoint('runs/pretrain_debug/clean_phi-1_5_lr1e-5_len1024_4_1_4', tag = 40000)
    
            
    print_gpu_utilization()


if __name__ == '__main__':
    main()