"""
Load the CUAD supervised fine-tuning data.
"""
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Any

class CUAD_SFT(Dataset):
    """
    Dataset for self supervised tuning. Add eos token to tokenization result.
    Args:
        path: SFT prompt data path. Each line is a json contains 'prompt' field
    """
    def __init__(
        self, path, tokenizer: PreTrainedTokenizer, max_length, small = False
    ):
        with open(path) as f:
            self.data = [json.loads(k) for k in f]
        self.small = small
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.cache = {} # idx to tokenization results
    
    def __len__(self):
        if self.small:
            return 200
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        sample = self.process(self.data[idx])
        self.cache[idx] = sample
        return sample
    
    def process(self, prompt_data):
        tokenizer = self.tokenizer
        enc = tokenizer(prompt_data['prompt'], max_length = self.max_length)
        if enc['input_ids'][-1] != tokenizer.eos_token_id:
            enc['input_ids'].append(tokenizer.eos_token_id)
            for k,v in enc.items():
                if k != 'input_ids':
                    v.append(v[-1])
        return {k:v for k,v in enc.items()}

class SFT_Padding:
    """Collator to pad SFT data to max length"""
    def __init__(self, pad_token_id, pad_side = 'right', pad_to_max_len = None):
        self.pad_token_id = pad_token_id
        self.pad_side = pad_side
        self.pad_to_max_len = pad_to_max_len
    
    def __call__(self, samples):
        keys = samples[0].keys()
        # convert list of sample to dict from key to list of values
        batch = dict(
            zip(
                keys,
                zip(*[[sample[k] for k in keys] for sample in samples])
            )
        )
        batch_tensors = {}
        for k,v in batch.items():
            pad_value = self.pad_token_id if k == 'input_ids' else 0
            batch_tensors[k] = torch.tensor(self.pad_list(v, pad_value))
        return batch_tensors
    
    def pad_list(self, values: List[List[Any]], pad_value):
        max_len = max([len(k) for k in values])
        if self.pad_to_max_len is not None:
            max_len = max(max_len, self.pad_to_max_len)
        pad_values = [self.pad(v, max_len, pad_value) for v in values]
        return pad_values
    
    def pad(self, L, max_len, pad_value):
        padded_v = [pad_value for _ in range(max_len - len(L))]
        if self.pad_side == 'left':
            return [*padded_v, *L]
        elif self.pad_side == 'right':
            return [*L, *padded_v]
        else:
            raise ValueError(f'Unexpected pad_side={self.pad_side}')