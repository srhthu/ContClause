"""
The pipeline function to infer a qa model and save outputs.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from tqdm import tqdm, trange

import transformers
from transformers import (
    AutoModelForQuestionAnswering,
)

from .trainer.train_only_ddp import Trainer_Onlytrain_DDP, TrainingArgs
from .trainer.utils import get_smart_optimizer
from .data_loader.dl_qa import CUAD_QA_Collator

MODEL_OUTPUT_FILE = 'model_outputs.pkl'

class CUAD_Trainer(Trainer_Onlytrain_DDP):
    def compute_preds(self, model, batch):
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids']
        }
        outs = model(**inputs)
        return {
            'start_logits': outs['start_logits'],
            'end_logits': outs['end_logits']
        }

def infer_and_save(ckpt_dir, features, trainer_config, save_path = None):
    if save_path is None:
        # save to the ckpt path
        save_path = Path(ckpt_dir) / MODEL_OUTPUT_FILE

    model = AutoModelForQuestionAnswering.from_pretrained(ckpt_dir)

    trainer = CUAD_Trainer(
        config = trainer_config,
        model = model,
        collate_fn = CUAD_QA_Collator(),
    )

    all_preds, _ = trainer.eval_loop(features)

    print('Finish infer')
    
    keys = list(all_preds.keys())
    # print(f'Output keys: {keys}')
    results = [
        {k:all_preds[k][i].tolist() for k in keys} 
        for i in range(len(all_preds[keys[0]]))
    ]

    print(f'Save results to {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', help = 'path to save results')
    parser.add_argument('--ckpt', help = 'only infer on one checkpoint')
    parser.add_argument('--exp_dir', help = 'dir contain multiple checkpoints')
    parser.add_argument('--features')
    parser.add_argument('--device_batch_size', type = int, default = 4)
    
    args = parser.parse_args()
    

    features = pickle.load(open(args.features, 'rb'))

    config = TrainingArgs(
        device_batch_size = args.device_batch_size,
    )

    ckpt_dir_list = []
    if args.exp_dir is not None:
        ckpt_dir_list = [str(k) for k in Path(args.exp_dir).glob('checkpoint-*')]
    else:
        ckpt_dir_list = [args.ckpt]
    
    for ckpt in ckpt_dir_list:
        save_path = args.save_path if args.save_path is not None \
                        else Path(ckpt) / 'model_outputs.pkl'
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        print(f'Infer checkpoint: {ckpt}, save results to {save_path}, exist = {Path(save_path).exists()}')
        infer_and_save(ckpt, features, config, save_path = save_path)
    

if __name__ == '__main__':
    main()