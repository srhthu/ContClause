"""
Debug with tokenizer
"""
# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
# %%
tk = AutoTokenizer.from_pretrained('gpt2')
# %%
len(tk)
# %%
tk.add_special_tokens({'pad_token': '[PAD]'})
# %%
len(tk)
# %%
model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype = torch.float16)
# %%
model.resize_token_embeddings(len(tk))
# %%
model.get_input_embeddings().weight[-1]
# %%
token_embs = model.get_input_embeddings()
type(token_embs.weight.data)
# %%
_ = model.cuda()
# %%
input_ids = torch.randint(10, 1000, (2,52)).cuda()
# %%
out = model(input_ids = input_ids)
# %%
model.transformer.wpe.weight.shape
# %%
model.transformer.h[6].attn.c_attn.weight.dim()

# %%
lin = nn.Linear(2,3)
n_ps = list(lin.named_parameters())
para = n_ps[0][1]
prev_p = para.clone().detach()
print(prev_p)
lin.weight.data *= -1
print(para)
# print(list(lin.named_parameters()))
# %%
