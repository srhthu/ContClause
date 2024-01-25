"""
Debug with tokenizer
"""
# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
# %%
model_name = "google/flan-t5-base"
tk = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# %%
type(model)
# %%
len(tk)
# %%
model
# %%
tk.special_tokens_map
# %%
tk.convert_tokens_to_ids([tk.pad_token])
# %%
input_ids = tk.encode('A step by step recipe to make pasta\n')
print(tk.convert_ids_to_tokens(input_ids))
# %%
model.config.decoder_start_token_id
# %%
