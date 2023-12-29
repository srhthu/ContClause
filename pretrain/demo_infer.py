"""
Load saved models and do inference.
"""
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# %%
base_model_name = 'microsoft/phi-1_5'
config = AutoConfig.from_pretrained(base_model_name, trust_remote_code = True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code = True, torch_dtype = torch.float16)
# %%
model_path = 'runs/cuad_phi-15_zero1/model_state/step-153600/pytorch_model.bin'
model = load_checkpoint_and_dispatch(model, model_path, 
                                    #  torch_dtype = torch.float16,
                                     device_map = 'auto')
# %%
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# %%
inputs = tokenizer('''
   Write a contract about joint venture of company A and company B. Start at 2014, Company A should provide Company B with human labors, while
   ''', return_tensors="pt", return_attention_mask=False)
inputs = {k:v.cuda() for k,v in inputs.items()}
outputs = model.generate(**inputs, max_length=200, temperature = 0.8)
text = tokenizer.batch_decode(outputs)[0]
print(text)
# %%
print(repr(text))
# %%
