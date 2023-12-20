"""
A demo of pretraining using deepspeed.

Aim:
    providing a toy example
    investigating memory usage of different input seq length.
"""
import argparse
import sys
import json
import torch
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
import accelerate
from accelerate import PartialState, Accelerator
import torch.nn as nn
from torch.utils.data import DataLoader
from pynvml import *

from pretrain.data_loader import FileLoader

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def build_model(name_or_path, torch_dtype, local_rank):
    config = AutoConfig.from_pretrained(name_or_path)
    kws = dict(
        torch_dtype = torch_dtype,
        # device_map = 'auto'
        device_map = local_rank
    )
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, **kws)
    else:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kws)
    return model

def build_lora_model(model):
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

def main_log(msg):
    state = PartialState()
    if state.local_process_index == 0:
        print(msg)

def process_log(msg):
    """
    Prepend with each process's index
    """
    state = PartialState()
    print(f'[Process {state.local_process_index}] {msg}')

def train_with_deepspeed(args, model, dataloader, ignore_index):
    ds_config = json.load(open(args.ds_config))
    ds_config['train_micro_batch_size_per_gpu'] = args.device_bs
    ds_config['gradient_accumulation_steps'] = args.grad_acc_steps
    if 'train_batch_size' in ds_config:
        _ = ds_config.pop('train_batch_size')
    
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
    for step in range(args.max_steps):
        batch = next(data_iter)
        loss = compute_loss(model, batch, ignore_index = ignore_index)
        loss_host += loss.detach().cpu().item()
        # Modified. Replace loss.backward()
        model.backward(loss)
        if (step+1) % args.grad_acc_steps == 0:
            # Modified. Replace optimizer.step()
            model.step()

            if (step + 1) % args.log_steps == 0:
                loss_host /= (args.log_steps * args.grad_acc_steps)
                main_log(f'Step {step+1} loss: {loss_host:.4f}')
                loss_host = 0.0

def train_ddp_accelerate(args, accelerator: Accelerator, model, dataloader, ignore_index):
    """
    Train with DDP.
    """
    data_iter = iter(dataloader)
    loss_host = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    
    model, optimizer = accelerator.prepare(model, optimizer)
    
    model.train()
    for step in range(args.max_steps):
        batch = next(data_iter)
        # print(f'input shape: {batch["input_ids"].shape}', flush = True)
        # exit()
        with accelerator.accumulate(model):
            loss = compute_loss(model, batch, ignore_index = ignore_index)
        loss_host += loss.detach().cpu().item()
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if (step+1) % (args.grad_acc_steps * args.log_steps) == 0:
            loss_host /= (args.log_steps * args.grad_acc_steps)
            main_log(f'Step {step+1} loss: {loss_host:.4f}')
            loss_host = 0.0

def print_device(named_parameters):
    for name, para in named_parameters:
        print(f'{name}: {para.device}')

def smart_resize_embeddings(tokenizer, model: PreTrainedModel):
    """
    Resize the model input embeddings and synchronous the new token embeddings across devices
    """
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) == old_vocab_size:
        # do not need resize embedding
        return
    num_new_tokens = len(tokenizer) - old_vocab_size
    main_log(f"Resize to add {num_new_tokens} new tokens")
    model.resize_token_embeddings(num_new_tokens)
    token_emb = model.get_input_embeddings()

    new_embs_data = token_emb.weight.data[-num_new_tokens:]
    accelerate.utils.broadcast(new_embs_data, from_process = 0)
    process_log(f'{token_emb.weight.data[-1,:10]}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--dtype', choices = ['fp16', 'bf16'], default = 'fp16')
    parser.add_argument('--ds_config')
    parser.add_argument('--lora', action = 'store_true')
    parser.add_argument('--max_steps', type = int, default = 200)
    parser.add_argument('--log_steps', type = int, default = 5)
    parser.add_argument('--device_bs', type = int, default = 2)
    parser.add_argument('--grad_acc_steps', type = int, default = 2)
    args = parser.parse_args()

    torch_dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_acc_steps)
    state = PartialState()

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code = True)
    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load dataset
    dataset = FileLoader(
        '/storage_fast/rhshui/workspace/datasets/legal/cuad_contracts', 
        tokenizer = tokenizer, max_length = 52
    )
    def collate_fn(samples):
        # For each key, gather the values of samples
        batched = dict(zip(samples[0].keys(),
                           zip(*[[sample[k] for k in sample] for sample in samples])))
        batched = {k: torch.tensor(v) for k,v in batched.items()}
        return batched
    dataloader = DataLoader(dataset, batch_size = args.device_bs, collate_fn = collate_fn)
    dataloader = accelerator.prepare(dataloader)

    # Build model
    model = build_model(args.model_path, torch_dtype_map[args.dtype], state.local_process_index)

    smart_resize_embeddings(tokenizer, model)

    # Lora
    if args.lora:
        model = build_lora_model(model)

    print_gpu_utilization()
    time.sleep(4)
    train_ddp_accelerate(args, accelerator, model, dataloader, ignore_index = tokenizer.pad_token_id)
    
    # train_with_deepspeed(args, model, dataloader, ignore_index=tokenizer.pad_token_id)

    

    # Start training
    
            
    print_gpu_utilization()


if __name__ == '__main__':
    main()