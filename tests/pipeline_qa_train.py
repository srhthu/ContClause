"""
Test the qa training function (train_only)
"""
# %%
import json
from pathlib import Path
from importlib import reload
from typing import Dict
import torch
from transformers import AutoTokenizer
import context
import numpy as np
import re
import pickle

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    RobertaForQuestionAnswering,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)

import cont_gen
import cont_gen.trainer.train_only_ddp as train_only_ddp
import cont_gen.data_loader.dl_qa as dl_qa
# %%
# reload the module if changed
reload(cont_gen)
reload(train_only_ddp)
reload(dl_qa)
from cont_gen.data_loader.dl_qa import CUAD_QA_Collator
from cont_gen.data_process.build_qa_feature import (
    create_examples, convert_features, convert_token_char_map,
    process_features
)
from cont_gen.trainer.utils import get_smart_optimizer
from cont_gen.trainer.train_only_ddp import TrainingArgs, Trainer_Onlytrain_DDP
# %%
model = AutoModelForQuestionAnswering.from_pretrained('roberta-base')
features = pickle.load(open('../data/qa_features/qa_roberta.pkl', 'rb'))
# %%
examples = pickle.load(open('../data/examples.pkl', 'rb'))
id2doc = pickle.load(open('../data/doc/doc_id_text.pkl', 'rb'))
# %%
optimizer = get_smart_optimizer(model, 1e-4, weight_decay=0.0)
# %%
class CUAD_Trainer(Trainer_Onlytrain_DDP):
    def compute_loss(self, model, batch):
        outs = model(**batch)
        return {'loss': outs['loss']}
# %%
config = TrainingArgs(
    batch_size= 16,
    device_batch_size=16,
    num_epoch = 3,
    logging_steps = 10,
    save_steps = 50
)
trainer = Trainer_Onlytrain_DDP(
    config = config,
    model = model,
    train_dataset = features[:3000],
    collate_fn = CUAD_QA_Collator(),
    output_dir = '../runs/debug',
    optimizer = optimizer
)
# %%
trainer.train()
# %%
