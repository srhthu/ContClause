"""Debug SFT training"""
# %%
import json
from pathlib import Path
from importlib import reload
from typing import Dict
from tqdm import tqdm
import numpy as np
import re
import pickle

import context

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    RobertaForQuestionAnswering,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoTokenizer,
    PhiForCausalLM,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from accelerate import Accelerator

import cont_gen
import cont_gen.trainer.train_only_accelerate as train_only_accelerate
import cont_gen.data_loader.cuad_prompt as dl_cuad
import cont_gen.trainer.utils_dist as utils_dist
import cont_gen.trainer.utils as tr_utils
# %%
reload(tr_utils)
reload(cont_gen)
reload(utils_dist)
reload(train_only_accelerate)
reload(dl_cuad)
from cont_gen.data_loader.cuad_prompt import CUAD_SFT, SFT_Padding, CUAD_SFT_Seq2Seq
# %%
phi_tk = AutoTokenizer.from_pretrained('/storage_fast/rhshui/llm/ms-phi-1_5')
print(phi_tk.special_tokens_map)
gpt2_tk = AutoTokenizer.from_pretrained('gpt2')
print(gpt2_tk.special_tokens_map)
t5_tk = AutoTokenizer.from_pretrained('google/flan-t5-base')
# %%
dataset = CUAD_SFT_Seq2Seq('../data/cuad_prompts/train_prompts_quest.jsonl', 
                   t5_tk, max_length = 1200, 
                   labels_on_full= False, small = True)
# %%
# test collate function
tokenizer = t5_tk
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
collator = SFT_Padding(tokenizer.pad_token_id, pad_side = 'right')
batch = collator([dataset[i] for i in range(5)])
for k,v in batch.items():
    print(f'{k}: {v.shape}')
# %%
gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
gpt2_model.resize_token_embeddings(len(gpt2_tk))
# %%
trained_gpt2 = AutoModelForCausalLM.from_pretrained('../runs/genqa/gpt2_quest_ds_2/checkpoint-2090')
# %%
phi_model = AutoModelForCausalLM.from_pretrained(
    '/storage/rhshui/workspace/contract_review/runs/debug_genqa/phi-15_ds/checkpoint-50', 
    torch_dtype = 'auto', 
    device_map = {'':0}, 
    trust_remote_code = True
)
# %%
input_ids = batch['input_ids'][:1,:180].cuda()
r = phi_model.generate(input_ids, max_new_tokens = 100, temperature = 0.8, do_sample = True)
# %%
print(phi_tk.batch_decode(input_ids)[0])
print(phi_tk.batch_decode(r)[0])
# %%
optimizer = torch.optim.AdamW(gpt2_model.parameters(), lr = 1e-4, weight_decay=0.01)
accelerator = Accelerator(gradient_accumulation_steps=2)
args = train_only_accelerate.TrainingArgs_Basic(
    device_batch_size=2,
    grad_acc_steps=2,
    max_epochs=3,
    logging_steps = 5,
    save_epochs = 1,
    save_total_limit=2
)
# %%
class CLM_Trainer(train_only_accelerate.Trainer_Basic):
    def compute_loss(self, model, batch):
        loss = tr_utils.compute_clm_loss_with_ignore(model, batch, ignore_index = gpt2_tk.pad_token_id)
        return {'loss': loss}

trainer = CLM_Trainer(
    args, gpt2_model, dataset, optimizer = optimizer,
    accelerator=accelerator, collate_fn = collator, output_dir = '../runs/debug_genqa/gpt2_4gpu_ddp'
)
# %%
trainer.train()
# %%
