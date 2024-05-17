"""
Supervised Finetuning with source and target data.
"""
import argparse
from pathlib import Path
import os
import json
import torch
from transformers import (
    AutoTokenizer, PreTrainedTokenizer
)
from peft import (
    LoraConfig
)
from accelerate import PartialState, Accelerator

from cont_gen.trainer.utils_dist import initialize_accelerator, DistLogger
from cont_gen.data_loader.cuad_prompt import CUAD_SFT, SFT_Padding, CUAD_SFT_Seq2Seq
from cont_gen.data_loader.cuad_sft import CUAD_SFT_Cached
from cont_gen.utils.model_utils import build_hf_or_peft_model, smart_resize_embeddings
from cont_gen.trainer.utils import get_smart_optimizer, compute_clm_loss_with_ignore
from cont_gen.trainer.train_only_accelerate import Trainer_Basic, TrainingArgs_Basic
from cont_gen.model.loss import LM_Simple_Feed

def load_train_dataset(args, tokenizer: PreTrainedTokenizer):
    """Cache in the save folder"""
    is_seq2seq = ('t5' in args.base_model)
    dataset = CUAD_SFT_Cached(
        args.data_path,
        tokenizer,
        is_seq2seq = is_seq2seq,
        is_chat = args.is_chat,
        cache_dir = Path(args.data_path).parent / 'cache',
        max_src_length = args.max_length,
        max_tgt_length = args.max_length,
        labels_on_full = args.labels_on_full,
        is_test = False,
        small = args.debug
    )
    return dataset

def build_model(args, accelerator, logger):
    """Build model, handle lora, quantization and gradient checkpointing"""
    if args.lora:
        if args.lora_all_linear:
            target_modules = 'all-linear'
        else:
            target_modules = args.__dict__.get('lora_target_modules', None)
        peft_config = LoraConfig(
            r = args.lora_r, 
            lora_alpha = args.lora_alpha,
            target_modules = target_modules,
            lora_dropout=args.lora_dropout,
            bias = "none"
        )
        # task_type will be configured later
    else:
        peft_config = None
    
    model = build_hf_or_peft_model(
        args.base_model, accelerator, args.dtype, 
        quantization=args.quantization,
        peft_config=peft_config
    )

    # load saved state dict
    if args.saved_model:
        logger.log(f'Load saved model parameters from: {args.saved_model}')
        with open(args.saved_model, 'rb') as f:
            state_dict = torch.load(f, map_location = accelerator.device)
        model.load_state_dict(state_dict)
        del state_dict
    
    # gradient checkpointing
    if args.gradient_checkpointing:
        logger.log(f'Enable gradient checkpointing...')
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    return model

def get_parser():
    parser = argparse.ArgumentParser()
    # run settings
    parser.add_argument('--output_dir')
    parser.add_argument('--ds_config')
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--total_batch_size', type = int)

    # data args
    parser.add_argument('--data_path', help = 'prompt file path')
    parser.add_argument('--max_length', type = int, default = None, 
                        help = 'max length for both source and target text.')
    parser.add_argument('--labels_on_full', action = 'store_true')
    parser.add_argument('--is_chat', action = 'store_true', 
                        help = 'whether apply chat template to data')

    # model args
    parser.add_argument('--base_model')
    parser.add_argument('--saved_model', help = 'path of saved model state dict',
                        nargs='?', const = None, default = None)
    parser.add_argument('--dtype', choices = ['fp16', 'bf16', 'fp32'], default = 'bf16')
    parser.add_argument('--quantization', action = 'store_true', help = 'quantize to 4bit')
    parser.add_argument('--gradient_checkpointing', action = 'store_true', 
                        help = 'enable gradient checkpointing')
    parser.add_argument('--lora', action = 'store_true', help = 'use lora adapter')
    parser.add_argument('--lora_r', type = int, default=16)
    parser.add_argument('--lora_alpha', type = int, default=16)
    parser.add_argument('--lora_all_linear', action = 'store_true', 
                        help = 'add lora to all linear layers')
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

    # We should set the env var of deepspeed before init partialstate
    if args.ds_config is not None:
        os.environ['ACCELERATE_USE_DEEPSPEED'] = 'true'

    # Initialize dist. environment
    state = PartialState()
    ## Calculate grad_acc_steps as suggested.
    if args.grad_acc_steps is None:
        args.grad_acc_steps = int(args.total_batch_size / args.device_batch_size / state.num_processes)
    
    accelerator, ds_config = initialize_accelerator(
        args.ds_config, args.device_batch_size, args.grad_acc_steps
    )

    # Get logger
    log_file = None if args.output_dir is None else str(Path(args.output_dir) / 'log.txt')
    logger = DistLogger(file = log_file)

    ## log arguments
    args_str = json.dumps(args.__dict__, indent = 4, ensure_ascii=False)
    if args.output_dir:
        with open(Path(args.output_dir) / 'args.json', 'w') as f:
            f.write(args_str)
    logger.log(f'Training args:\n {args_str}')

    logger.log(f'Distributed type: {accelerator.state.distributed_type}')

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code = True)
    if "pad_token" not in tokenizer.special_tokens_map:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # this is not good
        tokenizer.pad_token = tokenizer.eos_token
    
    # Build dataset
    is_seq2seq = ('t5' in args.base_model)
    dataset = load_train_dataset(args, tokenizer)

    ## data collate fn
    pad_args = {'pad_side': 'right'}
    if args.debug:
        pad_args['pad_to_max_len'] = args.max_length if is_seq2seq else args.max_length*2
    collate_fn = SFT_Padding(tokenizer.pad_token_id, **pad_args)

    # Build model
    model = build_model(args, accelerator, logger)

    ## resize model embedding if necessary
    # smart_resize_embeddings(tokenizer, model, logger)
    
    accelerator.wait_for_everyone()


    ## build optimizer
    optimizer = get_smart_optimizer(model, args.lr, args.weight_decay)

    # Build trainer
    ## training args
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
    assert tr_args.total_batch_size == args.total_batch_size

    ## Trainer
    trainer = Trainer_Basic(
        tr_args, model, dataset, optimizer, accelerator,
        ds_config = ds_config,
        collate_fn = collate_fn,
        compute_loss_fn = LM_Simple_Feed(),
        output_dir = args.output_dir,
        logger = logger
    )

    trainer.train()

if __name__ == '__main__':
    main()