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

class CachedDataset(Dataset):
    SMALL_NUM = 200
    """
    Support data caching and load small part.

    Attributes:
        cache_name
        data: (List[Any]) raw data
        samples: (List[dict]) processed data that will be cached
    """
    def __init__(self, cache_dir:Optional[str] = None, small:bool = False):
        self.cache_dir = cache_dir
        self.small = small

        self.dist_process()
    
    def dist_process(self):
        """Load and process data in distributed environment"""
        state = PartialState()
        # For distributed training, let main process first load or save cache, 
        # then other process load from the cache.
        with state.local_main_process_first():
            self.load_or_process_data()

    def load_or_process_data(self):
        if self.cache_dir is None:
            return
        # check whether cache exists
        cache_path = Path(self.cache_dir) / self.cache_name
        if cache_path.exists():
            print(f'Load from cache: {cache_path}')
            self.samples = load_pickle(cache_path)
        else:
            data_to_process = self.data if not self.small else self.data[:self.SMALL_NUM]
            self.samples = [self.process(k) for k in tqdm(data_to_process, ncols = 80)]
            print(f'Write to cache: {cache_path}')
            cache_path.parent.mkdir(parents = True, exist_ok = True)
            save_pickle(self.samples, cache_path)
    
    def __len__(self):
        return 200 if self.small else len(self.data)
    
    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def cache_name(self):
        # Customize for sub class
        return 'cache_default.pkl'
    
    def process(self, data):
        raise NotImplementedError

class CUAD_SFT_Cached(CachedDataset):
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
    SMALL_NUM = 200
    def __init__(
        self, path, tokenizer: PreTrainedTokenizer,
        is_seq2seq: bool,
        cache_dir: Optional[str] = None,
        max_src_length: Optional[int] = None, 
        max_tgt_length: Optional[int] = None,
        labels_on_full = False,
        is_test = False,
        small = False,
    ):
        self.data_path = path
        self.data = list(filter(self.filter_func, load_jsonl(path)))

        self.tokenizer = tokenizer
        self.is_seq2seq = is_seq2seq
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.labels_on_full = labels_on_full
        self.is_test = is_test

        super().__init__(cache_dir, small)
        
    def filter_func(self, data)->bool:
        """Filter raw data. Customize for sub classes"""
        return True

    @property
    def cache_name(self):
        data_name = Path(self.data_path).name
        debug_str = 'debug_' if self.small else ''
        tk_name = self.tokenizer.name_or_path.rstrip('/').split('/')[-1]
        cache_name = f'cached_{debug_str}{data_name}_{tk_name}_v{self.version}.pkl'
        return cache_name
    
    @staticmethod
    def get_tokenize_args(max_len):
        if max_len:
            return {'truncation': True, 'max_length': max_len}
        else:
            return {}

    def process(self, prompt_data):
        """
        Process data for decoder-only and seq2seq model
        """
        tokenizer = self.tokenizer

        src_tk_args = self.get_tokenize_args(self.max_src_length)
        if self.is_seq2seq:
            # do not remove eos token
            src_enc = tokenizer(prompt_data['source'], **src_tk_args)
        else:
            # for decoder-only model, do not add eos token to source
            src_enc = tokenize_wo_eos(tokenizer, prompt_data['source'],
                                      **src_tk_args)

        tgt_tk_args = self.get_tokenize_args(self.max_tgt_length)
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

            # Handle training to append target to source
            if not self.is_test:
                input_ids = input_ids + tgt_enc.input_ids
                attention_mask = attention_mask + tgt_enc.attention_mask
                # add eos token id
                if input_ids[-1] != tokenizer.eos_token_id:
                    input_ids.append(tokenizer.eos_token_id)
                    attention_mask.append(attention_mask[-1])
            
            # label only contain target tokens if not labels_on_full
            src_len = len(src_enc.input_ids)
            labels = [*input_ids] if self.labels_on_full else \
                        [-100] * src_len + input_ids[src_len:]
            return {'input_ids': input_ids, 
                    'attention_mask': attention_mask, 
                    'labels': labels}

class CUAD_SFT_Filter_Type(CUAD_SFT_Cached):
    """
    Filter raw data based on the value of types

    Args:
        judge_type_fn
    """
    def __init__(self, *args, **kws):
        self.judge_type = kws.pop('judge_type_fn', lambda k: True)
        super().__init__(*args, **kws)
    
    def filter_func(self, data) -> bool:
        return self.judge_type(data['type'])

class CUAD_SFT_Test_Part(CUAD_SFT_Cached):
    """Return the sampled test set for fast evaluate"""
    def filter_func(self, data) -> bool:
        return data['type'] > 0