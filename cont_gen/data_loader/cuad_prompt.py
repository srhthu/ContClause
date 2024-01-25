"""
Load the CUAD supervised fine-tuning data.
"""
import json
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from accelerate import PartialState
from typing import List, Any

class CUAD_SFT(Dataset):
    """
    Dataset for self supervised tuning. Add eos token to tokenization result.
    Args:
        path: SFT prompt data path. Each line is a json contains 'prompt' field
        labels_on_full: If true, labels are full inputs; otherwise only target tokens
    """
    def __init__(
        self, path, tokenizer: PreTrainedTokenizer, 
        max_length, labels_on_full = False, 
        is_test = False,
        small = False
    ):
        with open(path) as f:
            self.data = [json.loads(k) for k in f]
        self.small = small
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels_on_full = labels_on_full
        self.is_test = is_test

        state = PartialState()
        self.samples = [self.process(self.data[i]) for i in 
                        tqdm(list(range(len(self))), ncols = 80, disable = not state.is_main_process)]
    
    def __len__(self):
        if self.small:
            return 200
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def process(self, prompt_data):
        tokenizer = self.tokenizer
        enc = tokenizer(prompt_data['prompt'], max_length = self.max_length)
        if self.is_test:
            self.remove_eos_token(enc)
        else:
            self.add_eos_token(enc)
        sample = {k:v for k,v in enc.items()}
        # labels for training
        source_len = self.find_source_len(prompt_data['prompt'], enc)
        labels = self.get_labels(enc['input_ids'], source_len)
        sample['labels'] = labels
        return sample

    def find_source_len(self, text, enc):
        """
        Find the source token length. If no target, return num_tokens - 1
        """
        match = re.search('Answer:\n', text)
        if match is None:
            print('Warning: no target text')
            return len(enc['input_ids']) - 1
        end_char_pos = match.span(0)[1] - 1
        end_token_pos = enc.char_to_token(end_char_pos)
        if end_token_pos is None:
            return len(enc['input_ids']) - 1
        return min(end_token_pos + 1, len(enc['input_ids']) - 1)
    
    def get_labels(self, input_ids, source_len):
        """Set source labels to -100 and make sure there is at least one target token"""
        if self.labels_on_full:
            return [*input_ids]
        else:
            source_len = min(len(input_ids) - 1, source_len)
            labels = [-100] * source_len + input_ids[source_len:]
        return labels

    def add_eos_token(self, enc):
        tokenizer = self.tokenizer
        if enc['input_ids'][-1] != tokenizer.eos_token_id:
            enc['input_ids'].append(tokenizer.eos_token_id)
            for k,v in enc.items():
                if k != 'input_ids':
                    v.append(v[-1])
    def remove_eos_token(self, enc):
        tokenizer = self.tokenizer
        if enc['input_ids'][-1] == tokenizer.eos_token_id:
            _ = enc['input_ids'].pop(-1)
            for k,v in enc.items():
                if k != 'input_ids':
                    _ = v.pop(-1)

class CUAD_SFT_Seq2Seq(CUAD_SFT):
    def process(self, prompt_data):
        match = re.search('Answer:\n', prompt_data['prompt'])
        split_idx = match.span(0)[1]
        src_text, tgt_text = prompt_data['prompt'][:split_idx], prompt_data['prompt'][split_idx:]
        src_enc = self.tokenizer(src_text, truncation = True, max_length=self.max_length)
        input_ids, attention_mask = src_enc.input_ids, src_enc.attention_mask
        if len(tgt_text) == 0:
            # test data
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        tgt_enc = self.tokenizer(tgt_text, truncation = True, max_length=self.max_length)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': tgt_enc.input_ids
        }


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
            if k == 'input_ids':
                pad_value = self.pad_token_id
            elif 'label' in k:
                pad_value = -100
            else:
                pad_value = 0
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