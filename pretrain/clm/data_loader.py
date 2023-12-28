from pathlib import Path
import re
import random
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Union, List

from transformers import PreTrainedTokenizer

def get_sorted_child(dir_path, keep):
    """
    Get the child file or dir and sort by name.

    Args:
        keep: choice among (all, file, dir)
    """
    child = list(Path(dir_path).glob('*'))
    # filter child type
    if keep == 'file':
        child = [k for k in child if k.is_file()]
    elif keep == 'dir':
        child = [k for k in child if k.is_dir()]
    elif keep != 'all':
        raise ValueError(f'keep should be all, file or dir')
    # sort by name
    child = sorted(child, key = lambda k: k.name)

    return child


def get_file_recursively(dir_list: Union[str, List[str]], depth):
    """
    Get all file path under dirs in dir_list with depth = depth.

    depth = 1 refers to the direct child files.
    """
    assert depth >= 1 and isinstance(depth, int)
    if isinstance(dir_list, str):
        dir_list = [dir_list]
    if depth == 1:
        file_list = [
            path for leaf_dir in dir_list 
                for path in get_sorted_child(leaf_dir, keep = 'file')
        ]
        return file_list
    
    else:
        # get all dirs of next level
        next_dirs = [
            next_dir for cur_dir in dir_list 
                for next_dir in get_sorted_child(cur_dir, keep = 'dir')
        ]
        return get_file_recursively(next_dirs, depth - 1)

class CUAD_FileLoader(IterableDataset):
    """
    Load txt files under the folder. Support random seeds to control order.

    The data_dir has two levels. The first level is sub dirs, the second level is txt files.
    """
    def __init__(self, data_dir, tokenizer, max_length, seed = None):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed

        self.status = {
            'index': None,
            'file_index': None,
            'file_name': '',
            'chunk_index': None
        } 
        # record the position of current sample
        # keys: index, file_index, file_name, chunk_index

        if not Path(self.data_dir).exists():
            raise ValueError(f'Data folder does not exist. {self.data_dir}')

        self.load_and_cache_files()
    
    def load_and_cache_files(self):
        """Get all file paths and save the relative paths to cache."""
        data_dir = Path(self.data_dir)
        seed_str = 'ordinal' if self.seed is None else f'seed{self.seed}'
        cache_path = data_dir / f'cache_all_file_path_{seed_str}.txt'

        if Path(cache_path).exists():
            all_files = [k.strip() for k in open(cache_path).readlines()]
        else:
            all_files = get_file_recursively(self.data_dir, 2)
            all_files = [Path(k).relative_to(self.data_dir) for k in all_files]

            if self.seed is not None:
                all_files = random.Random(self.seed).sample(all_files, k = len(all_files))
            
            with open(cache_path, 'w') as f:
                f.write('\n'.join([str(k) for k in all_files]))
        
        self.all_files = all_files

    def __iter__(self):
        status = self.status
        prev_file_index, prev_chunk_index = (
            status['file_index'], status['chunk_index']
        )

        for file_idx, file_path in enumerate(self.all_files):
            if (prev_file_index is not None) and (file_idx < prev_file_index):
                continue
            
            # prev last file or a new file
            doc = open(Path(self.data_dir) / file_path).read()

            self.status['file_index'] = file_idx
            self.status['file_name'] = str(file_path)
            
            for chunk_i, chunk_enc in enumerate(self.process_doc(doc)):
                if (
                    (prev_file_index is not None)
                    and (file_idx == prev_file_index) 
                    and chunk_i <= prev_chunk_index
                ):
                    # in previous last doc generated chunks
                    continue
                if status['index'] is None:
                    status['index'] = 0
                else:
                    status['index'] += 1
                self.status['chunk_index'] = chunk_i
                yield chunk_enc

    def set_status(self, saved_status):
        for k in ['index', 'file_index', 'file_name', 'chunk_index']:
            self.status[k] = saved_status[k]

    
    def process_doc(self, doc):
        """
        Return the tokenization result of chunks of one document.
        """
        doc = self.clean_text(doc)
        enc = self.tokenizer(
            doc, max_length = self.max_length, 
            truncation = True, padding = 'max_length',
            return_overflowing_tokens = True
        )
        
        ignore_keys = ['overflow_to_sample_mapping']

        for chunk in zip(*[enc[k] for k in enc.keys()]):
            enc_dict = dict(zip(enc.keys(), chunk))
            enc_dict = {k:v for k,v in enc_dict.items() if k not in ignore_keys}

            length = sum(enc_dict['attention_mask'])

            if length < 2:
                continue
            
            # enc_dict['length'] = length
            assert enc_dict is not None
            yield enc_dict

    def clean_text(self, doc):
        return doc

class CUAD_FileLoader_Clean(CUAD_FileLoader):
    """Remove extra space and line break"""
    def clean_text(self, doc):
        lines = doc.split('\n')
        pat = r'[ ]{2,}'
        lines = [re.sub(pat, ' ', k) for k in lines]
        return '\n'.join(lines)


if __name__ == '__main__':
    # Test code
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--tokenizer', default = 'gpt2')
    args = parser.parse_args()

    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    tk.add_special_tokens({'pad_token': '[PAD]'})

    dl = FileLoader(args.data_dir, tk, max_length = 128)

    count = 0
    batch = []
    for enc_dict in dl:
        count += 1
        print(dl.status)
        print(enc_dict)
        batch.append(enc_dict)
        if count > 10:
            break
    
    def collate_fn(samples):
        # For each key, gather the values of samples
        batched = dict(zip(samples[0].keys(),
                            zip(*[[sample[k] for k in sample] for sample in samples])))
        # batched = {k:torch.stack([torch.tensor(e) for e in v], dim = 0) for k,v in batched.items()}
        batched = {k: torch.tensor(v) for k,v in batched.items()}
        return batched
    
    batch = collate_fn(batch)
    for k,v in batch.items():
        print(f'{k}: {v.shape} {v.dtype}')