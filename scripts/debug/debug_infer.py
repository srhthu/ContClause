# Debug of infering of genqa phi model
# %%
import json
from pathlib import Path
import sys
import re
sys.path.insert(0, str(Path(__file__).parents[1].absolute()))
# %%
from tqdm import tqdm
import numpy as np
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import string
from importlib import reload
import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, GenerationConfig, 
    GenerationMixin, PreTrainedModel, AutoModelForCausalLM
)
# %%
import cont_gen
reload(cont_gen)
from cont_gen.model_utils import load_hf_model_from_checkpoint
from cont_gen.data_loader.cuad_prompt import CUAD_SFT, CUAD_SFT_Seq2Seq, SFT_Padding
from cont_gen.trainer.train_only_accelerate import Predictor
# %%
GPU_N = 2
# %%
# base_model = '/storage_fast/rhshui/llm/ms-phi-1_5'
base_model = 'google/flan-t5-large'
tk = AutoTokenizer.from_pretrained(base_model)
if 'pad_token' not in tk.special_tokens_map:
    tk.add_special_tokens({'pad_token': '[PAD]'})
# %%
model_path = '../runs/genqa/phi-15_quest_lr1e-5/checkpoint-12540'
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code = True,
    torch_dtype = torch.bfloat16,
    device_map = {'':GPU_N}
)
# %%
dataset = CUAD_SFT(
    '../data/cuad_prompts/test_prompts_quest.jsonl',
    tk, max_length = 768, small = False
)
print(dataset[0].keys())
# %%
input_ids = dataset[0]['input_ids'][:-1]
input_ids = torch.tensor(input_ids).unsqueeze(0).cuda(GPU_N)
print(input_ids.shape)
# %%
gen_config = GenerationConfig(
    max_new_tokens = 512, eos_token_id = tk.eos_token_id,
    pad_token_id = tk.pad_token_id
)
# %%
out = model.generate(input_ids, gen_config)
# %%
tk.decode(out[0])
# %%
collate_fn = SFT_Padding(tk.pad_token_id, pad_side = 'left')
batch = collate_fn([dataset[i] for i in range(4)])
# %%
out = model.generate(
    input_ids = batch['input_ids'][...,:-1].cuda(GPU_N),
    attention_mask = batch['attention_mask'][...,:-1].cuda(GPU_N),
    generation_config = gen_config
)
# %%
prompt_data = [json.loads(k) for k in open('../data/cuad_prompts/test_prompts_quest.jsonl')]
# %%
prompt_data[131394]
# %%
def find_source_len(text, tokenizer, max_length):
    """
    Find the source token length. If no target, return num_tokens - 1
    """
    match = re.search('Answer:\n', text)
    enc = tokenizer(text, max_length = max_length)
    if match is None:
        print('Warning: no target text')
        return len(enc['input_ids']) - 1
    end_char_pos = match.span(0)[1] - 1
    end_token_pos = enc.char_to_token(end_char_pos)
    return min(end_token_pos + 1, len(enc['input_ids']) - 1)
# %%
find_source_len(prompt_data[131395]['prompt'], tk, 768)
# %%
for i in tqdm(range(130000, 140000)):
    try:
        _ = find_source_len(prompt_data[i]['prompt'], tk, 768)
    except:
        print(i)
        break
# %%
prompt_data[131569]
# %%
enc = tk(prompt_data[131569]['prompt'], max_length = 768)
print(len(enc.input_ids))
# %%
