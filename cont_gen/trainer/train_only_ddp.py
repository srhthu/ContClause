"""
Trainer that only train the model, no evaluate, DDP setting.
"""

import os
import sys
import json
import logging
from tokenize import Name
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math
import numpy as np
from tqdm import tqdm
import random
import time
import shutil
from pathlib import Path
import re
from argparse import Namespace
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader
from transformers.data.data_collator import default_data_collator
from transformers.optimization import (
    AdamW,
    get_scheduler
)
from transformers.trainer_pt_utils import (
    get_model_param_count,
    nested_numpify,
    nested_concat
)
from accelerate import Accelerator, PartialState

from .utils import AverageTensors, number2str
# from cont_gen.data_loader.dl_qa import Data_Collator

@dataclass
class TrainingArgs:
    """
    General arguments for a trainer
    """
    batch_size: int = 16  # total batch size
    device_batch_size: int = None
    eval_batch_size: Optional[int] = None # per device
    # the gradient_accumulate_step will be inferred
    max_grad_norm = 3.0

    # flow control
    num_epoch: int = None
    max_steps: Optional[int] = None
    logging_steps: int = 10 
    
    save_steps: Optional[int] = None
    save_epochs: Optional[int] = None
    
    def __post_init__(self):
        state = PartialState()
        if self.eval_batch_size is None:
            self.eval_batch_size = self.device_batch_size
        # infer gradient accumulate step
        if self.device_batch_size * state.num_processes > self.batch_size:
            # decrease the actual device_batch_size
            self.device_batch_size, r = divmod(self.batch_size, state.num_processes)
            print(f'reset device_batch_size to {self.device_batch_size}')
            g_acc_step = 1
        else:
            g_acc_step, r = divmod(self.batch_size, self.device_batch_size * state.num_processes)
        assert r == 0, (
            f"Cannot solve gradient accumulation step. batch_size={self.batch_size},"
            f"device_batch_size={self.device_batch_size}, n_proc={state.num_processes}\n"
        )
        self.gradient_accumulation_steps = g_acc_step
        
class DistLogger:
    """Only log on the local main process"""
    def __init__(self, name, output_dir = None):
        self.logger = logging.getLogger(name)
        self.output_dir = output_dir
        if output_dir:
            Path(output_dir).resolve().mkdir(exist_ok=True, parents=True)
            self.file_hdl = logging.FileHandler(Path(output_dir) / 'logs.txt')
            self.logger.addHandler(self.file_hdl)
        else:
            self.file_hdl = None
        self.std_hdl = logging.StreamHandler()
        
        self.logger.setLevel(10)

        self.state = PartialState()

    def info(self, msg, disable_std = False):
        if self.state.is_local_main_process:
            if disable_std:
                self.logger.info(msg)
            else:
                self.logger.addHandler(self.std_hdl)
                self.logger.info(msg)
                self.logger.removeHandler(self.std_hdl)

class Trainer_Onlytrain_DDP:
    """
    Training models on one GPU. Eval by steps.

    Do not support DataParallel

    Attributes:
        - output_dir: if not None, should exist

    Training arguments:
        - eval_steps: iter number if positive else epoch number if negative
        - early_stop_patience: same as above. wait > patience then stop
    """
    PREFIX_CHECKPOINT_DIR = 'checkpoint'
    WEIGHTS_NAME = "pytorch_model.bin"
    OPTIMIZER_NAME = "optimizer.pt"
    SCHEDULER_NAME = "scheduler.pt"
    TRAINING_ARGS_NAME = "training_args.json"

    def __init__(
        self,
        config: TrainingArgs,
        model,
        train_dataset,
        collate_fn = None,
        output_dir = None,
        optimizer = None,
        scheduler = None
    ):  
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn if collate_fn else default_data_collator
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.output_dir = output_dir

        self.logger = DistLogger('mytrainer', output_dir)

        self.init_accelerator()
    
    def init_accelerator(self):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.config.device_batch_size, 
            shuffle = True, 
            drop_last = True,
            collate_fn = self.collate_fn
        )

        
    def train(self):
        config = self.config
        g_acc_step = config.gradient_accumulation_steps
        logger = self.logger
        self.log_args()

        # Prepare for DDP
        train_dl = self.get_train_dataloader()
        model, train_dl, optimizer, scheduler = self.accelerator.prepare(
            self.model, train_dl, self.optimizer, self.scheduler
        )

        # determin max steps
        iter_per_epoch = len(train_dl) / g_acc_step
        num_epoch = config.num_epoch
        if num_epoch is None:
            max_steps = config.max_steps
            num_epoch = math.ceil(max_steps / iter_per_epoch)
        else:
            max_steps = iter_per_epoch * num_epoch # num of optimization step

        # get save_steps
        if any([config.save_steps, config.save_epochs]):
            save_steps = config.save_steps or iter_per_epoch * config.save_epochs
        else:
            save_steps = None

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {max_steps * config.batch_size:,}")
        logger.info(f"  Steps per epoch: {iter_per_epoch}")
        logger.info(f"  Num Epochs = {num_epoch:,}")
        logger.info(f"  Batch size per device = {config.device_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {config.batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {g_acc_step}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        # variables
        self.global_step = 0
        total_batched_samples = 0
        tr_metrics = AverageTensors()
        # start training
        model.train()
        training_bar = tqdm(
            total = max_steps, dynamic_ncols= True, 
            disable = not self.accelerator.is_local_main_process
        )
        for epoch in range(num_epoch):
            # epoch begin
            for step, batch in enumerate(train_dl):
                total_batched_samples += 1
                # forward and backward
                with self.accelerator.accumulate(model):
                    outputs = self.training_step(model, batch)
                tr_metrics.record(outputs)
        
                if total_batched_samples % g_acc_step == 0:
                    # optimization step
                    self.accelerator.clip_grad_norm_(
                        model.parameters(), 
                        config.max_grad_norm
                    )
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    model.zero_grad()
                    # step end
                    self.global_step += 1
                    training_bar.update(1)
                    
                    if self.global_step % config.logging_steps == 0:
                        # logging
                        tr_logs = tr_metrics.average()
                        tr_logs['step'] = self.global_step
                        tr_logs['epoch'] = epoch + step / len(train_dl)
                        self.log(tr_logs, tqdm_bar = training_bar)
                    
                    if save_steps and (self.global_step % save_steps == 0):
                        # evaluate
                        self.save_checkpoint()
                    # step end
            # epoch end
        # training end
        training_bar.close()
    
    def training_step(self, model, batch)->Dict[str, torch.Tensor]:
        """
        Prepare inputs, forward and backward.
        """
        outputs = self.compute_loss(model, batch)
        loss = outputs['loss']
        self.accelerator.backward(loss)
        return outputs
    
    def compute_loss(self, model, batch)-> Dict[str, torch.Tensor]:
        """
        Return a dict that mush contain loss
        """
        return model(**batch)
    
    def _log(self, msg: str, tqdm_bar = None):
        if not self.accelerator.is_local_main_process:
            return None
        if tqdm_bar is not None:
            tqdm_bar.write(msg)
        
        self.logger.info(msg, disable_std = (tqdm_bar is not None))

    def log(self, logs: Union[str, Dict[str, float]], tqdm_bar = None):
        if isinstance(logs, dict):
            logs = {k: number2str(v) for k,v in logs.items()}
        self._log(str(logs), tqdm_bar)

    def save_checkpoint(self):
        if not self.accelerator.is_local_main_process:
            return None
        
        ckpt_dir = Path(self.output_dir) / f'{self.PREFIX_CHECKPOINT_DIR}-{self.global_step}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_model(ckpt_dir)

        # Do not save optimizer to save disk space.
        # torch.save(self.optimizer.state_dict(), ckpt_dir / self.OPTIMIZER_NAME)
        # if self.scheduler:
        #     torch.save(self.scheduler.state_dict(), ckpt_dir / self.SCHEDULER_NAME)

    def save_model(self, output_dir: str):
        torch.save(self.model.state_dict(), Path(output_dir) / self.WEIGHTS_NAME)

    def log_args(self):
        self.logger.info('\n' + \
            json.dumps(self.config.__dict__, indent = 4, sort_keys = True))
    