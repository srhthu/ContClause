"""
The pipeline function to train a qa model.
"""
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from tqdm import tqdm, trange

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .trainer.train_only_ddp import Trainer_Onlytrain_DDP, TrainingArgs
from .trainer.utils import get_smart_optimizer
from .data_loader.dl_qa import CUAD_QA_Collator

class CUAD_Trainer(Trainer_Onlytrain_DDP):
    def compute_loss(self, model, batch):
        outs = model(**batch)
        return {'loss': outs['loss']}

def add_train_args(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--weight_decay', type = float, default = 0.)
    parser.add_argument('--num_epoch', type = int)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--device_batch_size', type = int, default = 16)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help = 'output path of the experiment')
    parser.add_argument('--base_model')
    parser.add_argument('--features')
    add_train_args(parser)
    
    args = parser.parse_args()

    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)

    features = pickle.load(open(args.features, 'rb'))

    optimizer = get_smart_optimizer(model, lr = args.lr, weight_decay=args.weight_decay)
    scheduler = None

    config = TrainingArgs(
        batch_size = args.batch_size,
        device_batch_size = args.device_batch_size,
        num_epoch = args.num_epoch,
        logging_steps = 20,
        save_steps = 500, 
    )

    trainer = Trainer_Onlytrain_DDP(
        config = config,
        model = model,
        train_dataset = features,
        collate_fn = CUAD_QA_Collator(),
        output_dir = args.output_dir,
        optimizer = optimizer,
        scheduler = scheduler
    )

    trainer.train()

if __name__ == '__main__':
    main()