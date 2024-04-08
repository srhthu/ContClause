"""
Trainer using accelerate and support DDP and Deepspeed

There supposed to be various trainers for different features
- Trainer_Basic: only support training, log and save
- Trainer_Basic_IterData: support iterative dataset
"""

import argparse
import sys
import re
import shutil
from pathlib import Path
import json
import numpy as np
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Mapping
import traceback

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedModel
from transformers.data.data_collator import default_data_collator
from peft import PeftModel
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
import accelerate
from accelerate import PartialState, Accelerator
from accelerate.utils import DummyOptim, DistributedType

from .utils_dist import is_state_initialized, initialize_accelerator, DistLogger
from .utils import nested_to_cpu

@dataclass
class TrainingArgs_Basic:
    """
    General arguments for a trainer to support training, log and save.
    """
    # batch_size: int = 16  # total batch size will be infered post init
    device_batch_size: int = None
    eval_batch_size: Optional[int] = None # per device
    grad_acc_steps: int = 1 # gradient accumulation steps
    # the gradient_accumulate_step will be inferred
    max_grad_norm = 3.0

    # flow control
    max_epochs: int = None
    max_steps: Optional[int] = None # micro steps, related to device_batch_size
    logging_steps: int = 10 
    
    save_steps: Optional[int] = None
    save_epochs: Optional[int] = None
    save_total_limit: Optional[int] = 3
    
    def __post_init__(self):
        if not is_state_initialized():
            print('Warning: PartialState has not been initialized.')
        state = PartialState()
        if self.eval_batch_size is None:
            self.eval_batch_size = self.device_batch_size
        # calculate total batch size
        self.total_batch_size = self.device_batch_size * self.grad_acc_steps * state.num_processes
        

class Trainer_Basic:
    """Basic trainer support training, log and save"""
    PREFIX_CHECKPOINT_DIR = 'checkpoint'
    WEIGHTS_NAME = "pytorch_model.bin"
    OPTIMIZER_NAME = "optimizer.pt"
    SCHEDULER_NAME = "scheduler.pt"
    TRAINING_ARGS_NAME = "training_args.json"

    def __init__(
        self,
        args: TrainingArgs_Basic,
        model,
        train_dataset,
        optimizer,
        accelerator: Optional[Accelerator] = None,
        ds_config = None,
        collate_fn = None,
        compute_loss_fn: Optional[Callable] = None,
        output_dir = None,
        scheduler = None,
        logger = None
    ):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn if collate_fn else default_data_collator
        self.compute_loss_fn = compute_loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.output_dir = output_dir
        self.logger = logger if logger else DistLogger(
            file = None if output_dir is None else Path(output_dir) / 'logs.txt'
        )

        if accelerator is None:
            self.accelerator, self.ds_config = initialize_accelerator(
                ds_config, 
                device_bs = args.device_batch_size, 
                grad_acc_steps = args.grad_acc_steps
            )
        else:
            self.accelerator, self.ds_config = accelerator, ds_config
    
    def get_next_batch(self, dataloader):
        """Get next from data iterator. If reach end, start a new iteration or exit"""
        args = self.args
        logger = self.logger
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.epoch_num += 1 # current finished epochs
            logger.log(f'Finish epoch num: {self.epoch_num}, total_steps: {self.total_step}')
            # save on epochs
            if args.save_epochs is not None and self.epoch_num % args.save_epochs == 0:
                self.save_checkpoint()
            if args.max_epochs is not None and self.epoch_num >= args.max_epochs:
                logger.log(f'Reach max_epoch={args.max_epochs}')
                return None
            self.data_iter = iter(dataloader)
            batch = next(self.data_iter)
        except Exception as e:
            # logger.log(traceback.format_exc())
            # logger.log(str(e))
            raise e
        return batch

    def train(self, dataloader = None):
        args = self.args
        accelerator = self.accelerator
        logger = self.logger
        # Build dataloader
        dataloader = dataloader if dataloader else DataLoader(
            self.train_dataset, 
            batch_size = args.device_batch_size, 
            collate_fn = self.collate_fn,
            shuffle = True,
            drop_last = True
        )
        # prepare model and optimize
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            self.model, self.optimizer, dataloader, self.scheduler
        )
        (
            self.wrap_model, self.wrap_optimizer, 
            self.wrap_dataloader, self.wrap_scheduler
        ) = model, optimizer, dataloader, scheduler

        logger.log('Finish prepare')
        logger.log(f'{type(model)}')

        model.train()

        self.logs = []
        loss_host = torch.tensor(0., device = accelerator.device)

        assert (args.max_steps is None) != (args.max_epochs is None), (
            'can only specify one of max_steps and max_epochs'
        )

        # determin max_steps
        self.total_step = 0
        self.epoch_num = 0
        self.data_iter = iter(dataloader)
        steps_per_epoch = len(dataloader) if not isinstance(dataloader.dataset, IterableDataset) else None
        if args.max_epochs is not None:
            if not isinstance(dataloader.dataset, IterableDataset):
                max_steps = len(dataloader) * args.max_epochs
            else:
                max_steps = sys.maxsize
        else:
            max_steps = args.max_steps
        
        # prepare tqdm bar
        bar_kws = {'ncols': 100, 'disable': not accelerator.is_main_process}
        if max_steps != sys.maxsize:
            bar_kws['total'] = max_steps
        bar = tqdm(**bar_kws)
        logger.add_tqdm(bar)

        # log training parameters
        train_info = (
            f'Start Training\n'
            f'\tDevice batch size: {args.device_batch_size}\n'
            f'\tGradient accumulation steps: {args.grad_acc_steps}\n'
            f'\tNum processes: {accelerator.num_processes}\n'
            f'\tTotal batch size: {args.total_batch_size}\n'
            f'\tMax training steps / steps per epoch: {max_steps} / {steps_per_epoch}\n'
        )
        logger.log(train_info)

        batch = self.get_next_batch(dataloader)
        while self.total_step < max_steps:
            if batch is None:
                break
            with accelerator.accumulate(model):
                outputs = self.compute_loss(model, batch)
            loss = outputs['loss']
            loss_host += loss.detach() # keep it on gpu
            
            # backward
            accelerator.backward(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            self.total_step += 1
            bar.update(1)

            self.maybe_log(loss_host)
            
            if (
                self.output_dir is not None
                and args.save_steps is not None 
                and self.total_step % args.save_steps == 0
            ):
                self.save_checkpoint()
            # If reach end of an epoch and save by epochs,
            #call get_next_batch will save checkpoints
            batch = self.get_next_batch(dataloader)

        bar.close()
        logger.remove_tqdm()
        logger.log("Finish logging")
        
    def compute_loss(self, model, batch):
        """Should return a dict containing loss"""
        if self.compute_loss_fn is None:
            return model(**batch)
        else:
            return self.compute_loss_fn(model, batch)
    
    def maybe_log(self, loss_host):
        args = self.args
        log_micro_steps = args.grad_acc_steps * args.logging_steps
        if self.total_step % log_micro_steps == 0:
            loss_host /= log_micro_steps # in-place div here
            # average across devices
            loss_host_ave = self.accelerator.gather(loss_host).mean().cpu().item()
            message = {'step': self.total_step, 'loss': loss_host_ave}
            self.logger.log(str(message))
            if self.accelerator.is_main_process:
                self.write_loss_log(message)
            # reset loss to zero. Use in-place operation here.
            loss_host -= loss_host
    
    def write_loss_log(self, message):
        if self.output_dir is None:
            return
        with open(Path(self.output_dir)/'loss_log.jsonl', 'a') as f:
            f.write(json.dumps(message, ensure_ascii=False) + '\n')

    def save_checkpoint(self):
        """
        Save the model state_dict. 
        For transformers PreTrained models, also use save_pretrained. Otherwise, only save the weights.
        """
        # only save the model
        if self.output_dir is None:
            return
        accelerator = self.accelerator
        ckpt_dir = Path(self.output_dir) / f'{self.PREFIX_CHECKPOINT_DIR}-{self.total_step}'
        self.logger.log(f'Save model to {ckpt_dir}')

        unwrapped_model = accelerator.unwrap_model(self.wrap_model)
        if isinstance(unwrapped_model, (PreTrainedModel, PeftModel)):
            unwrapped_model.save_pretrained(
                str(ckpt_dir),
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                safe_serialization=False
            )
        else:
            accelerator.save_model(
                self.wrap_model, 
                str(ckpt_dir), 
                max_shard_size="1GB"
            )
        self.accelerator.wait_for_everyone()
        self.rotate_checkpoints()
    
    def rotate_checkpoints(self):
        output_dir = self.output_dir
        if output_dir is None:
            return
        if self.args.save_total_limit is None:
            return
        glob_ckpts = list(Path(output_dir).glob(f'{self.PREFIX_CHECKPOINT_DIR}-*'))
        glob_ckpts = [str(x) for x in glob_ckpts if x.is_dir()]

        ordered_ckpts = []
        for path in glob_ckpts:
            regex_match = re.match(f".*checkpoint-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordered_ckpts.append((
                    int(regex_match.groups()[0]), path
                ))
        
        ordered_ckpts.sort(key = lambda k: k[0])

        n_ckpt_to_delete = max(0, len(ordered_ckpts) - self.args.save_total_limit)
        for _, checkpoint in ordered_ckpts[:n_ckpt_to_delete]:
            self.logger.log(f'Delete old checkpoint: {checkpoint}')
            shutil.rmtree(checkpoint, ignore_errors = True)


class Predictor:
    """
    Predict in distributed setting.
    """
    def __init__(self, accelerator = None, batch_size = 1, compute_preds_fn = None, collate_fn = None):
        self.accelerator = accelerator if accelerator else Accelerator()
        self.batch_size = batch_size
        self.compute_preds_fn = compute_preds_fn
        self.collate_fn = collate_fn if collate_fn else default_data_collator
    
    def predict(self, model, dataset):
        """
        Return predic results of each sample. Support DDP.
        """
        accelerator = self.accelerator
        # get dataloader
        dataloader = DataLoader(
            dataset, batch_size = self.batch_size, 
            shuffle = False, drop_last = False, 
            collate_fn = self.collate_fn
        )
        # prepare for distributed system
        dataloader = accelerator.prepare(dataloader)
        # Do not prepare the model but use the original one.
        model.cuda(accelerator.local_process_index)
        model.eval()

        bar = tqdm(
            total = len(dataloader), ncols= 100, 
            disable = not self.accelerator.is_local_main_process
        )
        
        all_preds = [] if accelerator.is_main_process else None # sample predictions
        for batch in dataloader:
            with torch.no_grad():
                preds = self.compute_preds_fn(model, batch)
            device_sample_preds = self.batch_result_to_samples(preds)
            
            gathered_preds = self.gather_object(device_sample_preds)
            if accelerator.is_main_process:
                for dev_preds in gathered_preds:
                    for sample_pred in dev_preds:
                        all_preds.append(nested_to_cpu(sample_pred))
            
            accelerator.wait_for_everyone()
            
            bar.update()

        return all_preds
        
    def batch_result_to_samples(self, batch_results):
        """Convert a batch result (dict or list) to sample result"""
        if isinstance(batch_results, Mapping):
            keys = list(batch_results.keys())
            sample_results = list(zip(*batch_results.values()))
            return [dict(zip(keys, r)) for r in sample_results]

        elif isinstance(batch_results, (list, tuple)):
            return list(zip(*batch_results))
        
    def gather_object(self, data):
        accelerator = self.accelerator
        output = [None] * accelerator.num_processes if accelerator.is_main_process else None
        dist.gather_object(data, output)
        # main process: [sample_preds, ...]
        # other process: None
        return output
            