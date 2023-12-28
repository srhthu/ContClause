"""
Given a model and num of gpus, find the maximum batch_size_per_gpu by trying from 1 to 2^n

Support training mode:
    DistributedDataParallel
    Deepspeed zero stages
"""

import argparse
import sys
from pathlib import Path
import json
import torch
import numpy as np
import time
from tqdm import tqdm
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
import traceback

from clm.prepare import (
    load_and_update_ds_config, get_hf_ds_plugin,
    build_model, build_lora_model_demo, initialize_accelerator
)
from clm.utils import (
    get_gpu_utilization, print_gpu_utilization, get_gpu_memory,
    DistLogger, smart_resize_embeddings, ParamChangeChecker
)

LR=1E-4
GRAD_ACC_STEPS=2
NUM_MACRO_STEPS = 10

class PseudoIterDataset(torch.utils.data.IterableDataset):
    LOW_ID = 100
    HIGH_ID = 1000
    def __init__(self, seq_len, total_step = 100):
        self.seq_len = seq_len
        self.total_step = total_step
    
    def __iter__(self):
        for _ in range(self.total_step):
            data = {
                'input_ids': torch.randint(self.LOW_ID, self.HIGH_ID, (self.seq_len,)),
                'attention_mask': torch.ones(self.seq_len)
            }
            yield data

def collate_fn(samples):
    # For each key, gather the values of samples
    batched = dict(zip(samples[0].keys(),
                        zip(*[[sample[k] for k in sample] for sample in samples])))
    batched = {k: torch.stack(v, dim = 0) for k,v in batched.items()}
    return batched

def train_loop(model, accelerator: Accelerator, dataloader, logger):
    # Build optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=LR)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    micro_steps = GRAD_ACC_STEPS * NUM_MACRO_STEPS
    bar = tqdm(total = micro_steps, ncols = 80, 
               disable = (accelerator.local_process_index != 0))
    
    logs = []
    prev_time = time.time()
    for i, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            batch['labels'] = batch['input_ids']
            loss = model(**batch).loss
        # backward
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # time log
        cur_time = time.time()
        step_time = cur_time - prev_time
        prev_time = cur_time
        logs.append({'step': i, 'time': step_time})
        bar.update()
    bar.close()
    
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--dtype', choices = ['fp16', 'bf16', 'fp32'], default = 'fp16')
    parser.add_argument('--ds_config')
    parser.add_argument('--max_device_batch_size', type = int, default = 512)
    parser.add_argument('--max_length', type = int, default = 512)

    # this argumentis not used. Just to be compatible with deepspeed launch
    parser.add_argument('--local_rank')

    args = parser.parse_args()

    torch_dtype_map = {
        'fp16': torch.float16, 
        'bf16': torch.bfloat16, 
        'fp32': torch.float32
    }

    # start search
    """
    Search possible device batch_size and return training speed and memory usage.

    Return:
        results: a list of results:
            device_batch_size
            step_time: average time of processing one micro batch
            samples_per_second: num of samples finished by each gpu per second
            gpu_memory: map from gpu_id to gpu_used memory in MB, only inlude used gpus
            success (bool): 
    """
    model = None
    
    device_bs = 1
    results = []
    while device_bs <= args.max_device_batch_size:
        accelerator, ds_config = initialize_accelerator(
            args.ds_config, device_bs, GRAD_ACC_STEPS
        )
        logger = DistLogger()
        if device_bs == 1:
            logger.log_main(str(accelerator.state))
        state = PartialState()
        n_gpus = state.num_processes

        # Build model
        if model is None:
            model = build_model(args.model_path, torch_dtype_map[args.dtype], 
                        state.local_process_index, 
                        is_zero3 = False)
            print_gpu_utilization(logger)
        
        logger.log_main(f'Try device_bs={device_bs}')
        # Build data loader
        total_samples = n_gpus * device_bs * GRAD_ACC_STEPS * NUM_MACRO_STEPS
        dataset = PseudoIterDataset(args.max_length, total_samples)
        dataloader = DataLoader(dataset, batch_size = device_bs, collate_fn = collate_fn)

        try:
            logs = train_loop(model, accelerator, dataloader, logger)
        except Exception as e:
            # logger.log_main(traceback.format_exc())
            logger.log_main(str(e))
            result = {'device_batch_size': device_bs, 'success': False}
        else:
            # exclude the first step
            ave_step_time = np.mean([k['time'] for k in logs[1:]])
            samples_per_second = device_bs / ave_step_time
            
            result = {
                'device_batch_size': device_bs,
                'step_time': ave_step_time,
                'samples_per_second': samples_per_second,
                'gpu_memory': get_gpu_memory(),
                'success': True,
            }
        results.append(result)
        # logger.log_main(str(result))
        if not result['success']:
            break
        else:
            device_bs *= 2
    for r in results:
        logger.log_main(str(r))


if __name__ == '__main__':
    main()