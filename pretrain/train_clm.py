"""
Training code for causal language modeling
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

from clm.data_loader import CUAD_FileLoader, CUAD_FileLoader_Clean
from clm.utils import (
    get_gpu_utilization, print_gpu_utilization,
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

def collate_fn(samples):
    # For each key, gather the values of samples
    batched = dict(zip(samples[0].keys(),
                        zip(*[[sample[k] for k in sample] for sample in samples])))
    batched = {k: torch.tensor(v) for k,v in batched.items()}
    return batched


def train_accelerate(
    args, 
    accelerator: Accelerator, 
    model, 
    dataloader, 
    ignore_index,
    logger
):
    """
    Train with accelerate, including DDP and deepspeed setting.
    """
    loss_host = 0.0

    # Build optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    
    # prepare model and optimize
    model, optimizer = accelerator.prepare(model, optimizer)

    step_info, model, optimizer, dataloader = load_checkpoint(
        args.ckpt_dir, accelerator, model, optimizer, dataloader, logger)

    logger.log_main('Finish prepare')
    logger.log_main(f'{type(model)}')

    model.train()

    logs = []
    last_log_time = time.time()
    loss_host = torch.tensor(0., device = accelerator.device)

    data_iter = iter(dataloader)
    n_epoch = step_info['n_epoch']
    total_step = step_info['total_step']
    assert (args.max_steps is None) != (args.max_epochs is None), (
        'can only specify one of max_steps and max_epochs'
    )
    while True:
        try:
            batch = next(data_iter)
        # except StopIteration:
        except:
            # when finish a epoch, determin continue or stop
            n_epoch += 1
            if args.max_epochs is not None and n_epoch >= args.max_epochs:
                logger.log_main(f'Reach max_epoch = {args.max_epochs}')
                break
            
            logger.log_main(f'Finish epoch num: {n_epoch}, total_steps {total_step}')
            dataloader.dataset.reset()
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            loss = compute_loss(model, batch, ignore_index = ignore_index)
        loss_host += loss.detach() # keep it on gpu
        
        # backward
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_step += 1

        if (total_step) % (args.grad_acc_steps * args.log_steps) == 0:
            loss_host /= (args.log_steps * args.grad_acc_steps)
            # average across gpust
            loss_host_ave = accelerator.gather(loss_host).mean().cpu().item()
            dur = time.time() - last_log_time
            last_log_time = time.time()

            # prepare log
            file_index = dataloader.dataset.status['file_index']
            if file_index is None:
                file_index = 0
            loss_log = {
                'step': total_step, 'loss': loss_host_ave, 
                'file_index': file_index,
                'file_percent': round(
                    file_index / len(dataloader.dataset.all_files) * 100, 2),
                'time': dur}
            logger.log_main(str(loss_log))
            logs.append(loss_log)
            if accelerator.is_main_process:
                write_loss_log(args.output_dir, loss_log)

            loss_host -= loss_host
        
        if (total_step) % (args.save_steps) == 0:
            if args.output_dir is not None:
                ckpt_dir = Path(args.output_dir) / f'checkpoint-{total_step}'
                logger.log_main(f'Save checkpoint {total_step}, exist={ckpt_dir.exists()}')
                step_info = {'total_step': total_step, 'n_epoch': n_epoch}
                save_checkpoint(ckpt_dir, step_info, accelerator, model, optimizer, dataloader)

                # remove extra checkpoint
                rotate_checkpoints(args.output_dir, args.save_total_limit)
        
        if (args.max_steps is not None) and (total_step >= args.max_steps):
            break
    
    if args.max_epochs is not None:
        # save last step
        ckpt_dir = Path(args.output_dir) / f'checkpoint-{total_step}'
        if not ckpt_dir.exists():
            logger.log_main(f'Save checkpoint {total_step}')
            step_info = {'total_step': total_step, 'n_epoch': n_epoch}
            save_checkpoint(ckpt_dir, step_info, accelerator, model, optimizer, dataloader)

    return logs, model

def write_loss_log(run_dir, loss_log: dict):
    if run_dir is None:
        return
    with open(Path(run_dir) / 'loss_log.jsonl', 'a') as f:
        f.write(json.dumps(loss_log, ensure_ascii=False) + '\n')

def save_checkpoint(ckpt_dir, step_info, accelerator:Accelerator, model, optimizer, dataloader):
    ckpt_dir = Path(ckpt_dir)
    if accelerator.is_main_process:
        # print(f'rank {accelerator.process_index}, {ckpt_dir.exists()}')
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # save global step
        with open(ckpt_dir / 'step_info.json', 'w') as f:
            json.dump(step_info, f)
    
        # save dataset status
        dl_state = dataloader.dataset.status
        with open(Path(ckpt_dir) / 'dataset_state.json', 'w') as f:
            json.dump(dl_state, f, ensure_ascii=False)
    
    dist.barrier()
    # accelerator.save_model(model, ckpt_dir)
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(
    #     ckpt_dir,
    #     is_main_process=accelerator.is_main_process,
    #     save_function=accelerator.save,
    # )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        model.save_checkpoint(ckpt_dir)
    elif accelerator.distributed_type == DistributedType.MULTI_GPU:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            ckpt_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            torch.save(optimizer.state_dict(), str(Path(ckpt_dir)/'optimizer.pt') )
    else:
        raise ValueError(f'distributed type not supported: {accelerator.distributed_type}')
    
    dist.barrier()

def load_checkpoint(ckpt_dir, accelerator: Accelerator, model, optimizer, dataloader, logger = None):
    if ckpt_dir is None:
        return {'total_step':0,'n_epoch':0}, model, optimizer, dataloader
    ckpt_dir = Path(ckpt_dir)
    logger.log_main(f'Load checkpoint: {ckpt_dir}')
    # load total step
    with open(ckpt_dir / 'step_info.json') as f:
        step_info = json.load(f)
    logger.log_main(f'Finished: {step_info}')

    # Load dataset state
    with open(ckpt_dir / 'dataset_state.json') as f:
        dts_stat = json.load(f)
    logger.log_main(f'dataset status: {str(dts_stat)}')
    dataloader.dataset.set_status(dts_stat)

    # Load model and optimizer
    logger.log_main(f'Load model and optimizer')
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        model.load_checkpoint(ckpt_dir)
    elif accelerator.distributed_type == DistributedType.MULTI_GPU:
        model.module.load_state_dict(torch.load(ckpt_dir / 'pytorch_model.bin'))
        optimizer.load_state_dict(torch.load(ckpt_dir / 'optimizer.pt'))
    else:
        raise ValueError(f'unknown distributed type: {accelerator.distributed_type}')
    return step_info, model, optimizer, dataloader

def rotate_checkpoints(output_dir, save_total_limit):
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f'checkpoint-*') if x.is_dir()]

    ordered_ckpts = []
    for path in glob_checkpoints:
        regex_match = re.match(f".*checkpoint-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordered_ckpts.append((
                int(regex_match.groups()[0]), path
            ))
    
    ordered_ckpts.sort(key = lambda k: k[0])

    n_ckpt_to_delete = max(0, len(ordered_ckpts) - save_total_limit)
    for step, checkpoint in ordered_ckpts[:n_ckpt_to_delete]:
        shutil.rmtree(checkpoint, ignore_errors = True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--ckpt_dir', help = 'checkpoint dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--seed', type = int, help = 'seed for datset', default = 43)
    parser.add_argument('--model_path', default = 'microsoft/phi-1_5')
    parser.add_argument('--dtype', choices = ['fp16', 'bf16', 'fp32'], default = 'bf16')
    parser.add_argument('--ds_config')
    parser.add_argument('--lr', type = float, default = 1e-4)
    
    parser.add_argument('--max_epochs', type = int)
    parser.add_argument('--max_steps', type = int)
    parser.add_argument('--log_steps', type = int, default = 5, help = 'macro steps')
    parser.add_argument('--save_steps', type = int, default = 512000, help = 'micro steps')
    parser.add_argument('--device_bs', type = int, default = 2)
    parser.add_argument('--grad_acc_steps', type = int, default = 64)
    
    parser.add_argument('--save_total_limit', type = int, default = 5)
    
    parser.add_argument('--max_length', type = int, default = 1024)

    # to be compatible with deepspeed launch
    parser.add_argument('--local_rank')
    args = parser.parse_args()

    torch_dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

    # Initialize distribution environment
    accelerator, ds_config = initialize_accelerator(
        args.ds_config, args.device_bs, args.grad_acc_steps
    )
    state = PartialState()
    
    # Get logger
    log_file = None if args.output_dir is None else str(Path(args.output_dir) / 'log.txt')
    logger = DistLogger(file = log_file)

    # determin gradient_accumulation_step
    args.total_bs = args.device_bs * state.num_processes * args.grad_acc_steps

    # save args
    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents = True, exist_ok = True)
        with open(Path(args.output_dir) / 'args.json', 'w') as f:
            json.dump(args.__dict__, f)
    logger.log_main(f'Training args:\n {json.dumps(args.__dict__, indent = 4, ensure_ascii=False)}')

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code = True)
    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load dataset
    ds_cls = CUAD_FileLoader_Clean
    dataset = ds_cls(
        args.data_dir, tokenizer = tokenizer, 
        max_length = args.max_length,
        seed = args.seed
    )
    
    dataloader = DataLoader(dataset, batch_size = args.device_bs, collate_fn = collate_fn)
    dataloader = accelerator.prepare(dataloader)

    # Build model
    is_zero3 = False if accelerator.state.deepspeed_plugin is None else \
            accelerator.state.deepspeed_plugin.zero3_init_flag
    model = build_model(args.model_path, torch_dtype_map[args.dtype], 
                        state.local_process_index, 
                        is_zero3 = is_zero3)

    smart_resize_embeddings(tokenizer, model, logger)

    logs, engine = train_accelerate(
        args, accelerator, model, dataloader, 
        ignore_index = tokenizer.pad_token_id,
        logger = logger
    )


if __name__ == '__main__':
    main()