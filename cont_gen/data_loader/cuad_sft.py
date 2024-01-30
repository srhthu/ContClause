"""
Improved version of cuad_prompt.CUAD_SFT.

Support cache managment.
"""

import json
import re
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from accelerate import PartialState
from typing import List, Any, Optional

from cont_gen.utils import load_jsonl, load_pickle, save_pickle
from cont_gen.data_process.utils import tokenize_wo_eos

class CUAD_SFT_Cached(Dataset):
    """
    Process source and target sequence for supervised fine-tuning. Support caching.

    Args:
        path: path of jsonl data containing source and target
        tokenizer: transformer tokenizer
        is_seq2seq: (`bool`) denote encoder-decoder or decoder-only model
        max_length: max source token length
        max_target_length: max target token length
        labels_on_full: For decoder-only model, 
            True to train on both source and target. 
            Otherwise, train only on target tokens
        is_test: for test data, do not provide target sequence
        small: True for debug mode with few samples
    """
    version = "1.0"
    def __init__(
        self, path, tokenizer: PreTrainedTokenizer,
        is_seq2seq: bool,
        cache_dir: Optional[str] = None,
        max_length: Optional[int] = None, 
        max_target_length: Optional[int] = None,
        labels_on_full = False,
        is_test = False,
        small = False,
    ):
        self.data_path = path
        self.data = load_jsonl(path)

        self.tokenizer = tokenizer
        self.is_seq2seq = is_seq2seq
        self.cache_dir = cache_dir if cache_dir else './cache'
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.labels_on_full = labels_on_full
        self.is_test = is_test
        self.small = small

        state = PartialState()
        # For distributed training, let main process first load or save cache, 
        # then other process load from the cache.
        with state.local_main_process_first():
            self.load_or_process_data()

    def load_or_process_data(self):
        # check whether cache exists
        cache_path = Path(self.cache_dir) / self.cache_name
        if cache_path.exists():
            print(f'Load from cache: {cache_path}')
            self.samples = load_pickle(cache_path)
        else:
            self.samples = [self.process(k) for k in tqdm(self.data, ncols = 80)]
            print(f'Write to cache: {cache_path}')
            cache_path.parent.mkdir(parents = True, exist_ok = True)
            save_pickle(self.samples, cache_path)

    @property
    def cache_name(self):
        data_name = Path(self.data_path).name
        debug_str = 'debug_' if self.small else ''
        tk_name = self.tokenizer.name_or_path.split('/')[-1]
        cache_name = f'cached_{debug_str}{data_name}_{tk_name}_v{self.version}.pkl'
        return cache_name
    
    def __len__(self):
        return 200 if self.small else len(self.data)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def process(self, prompt_data):
        """
        Process data for decoder-only and seq2seq model
        """
        tokenizer = self.tokenizer

        src_tk_args = {} if self.max_length is None \
                    else {'truncation': True, 'max_length': self.max_length}
        src_enc = tokenize_wo_eos(tokenizer, prompt_data['source'],
                                   **src_tk_args)

        tgt_tk_args = {} if self.max_target_length is None \
                    else {'truncation': True, 'max_length': self.max_target_length}
        tgt_enc = tokenizer(prompt_data['target'], **tgt_tk_args)

        if self.is_seq2seq:
            return{
                'input_ids': src_enc.input_ids,
                'attention_mask': src_enc.attention_mask,
                'labels': tgt_enc.input_ids
            }
        else:
            input_ids = src_enc.input_ids
            attention_mask = src_enc.attention_mask
            if not self.is_test:
                input_ids += tgt_enc.input_ids
                attention_mask += tgt_enc.attention_mask
                if input_ids[-1] != tokenizer.eos_token_id:
                    input_ids.append(tokenizer.eos_token_id)
                    attention_mask.append(attention_mask[-1])
            src_len = len(src_enc.input_ids)
            labels = [*input_ids] if self.labels_on_full else \
                        [-100] * src_len + input_ids[src_len:]
            return {'input_ids': input_ids, 
                    'attention_mask': attention_mask, 
                    'labels': labels}