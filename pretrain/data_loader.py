from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from transformers import PreTrainedTokenizer

class FileLoader(IterableDataset):
    """
    Load txt files under the folder.

    The data_dir has two levels. The first level is sub dirs, the second level is txt files.
    """
    def __init__(self, data_dir, tokenizer, max_length):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.status = {} 
        # record the process
        # keys: sub_dir, file_name, chunk, total
    
    def __iter__(self):
        count = 0
        for sub_dir in Path(self.data_dir).glob("*"):
            self.status['sub_dir'] = str(sub_dir)
            for file in sub_dir.glob("*"):
                self.status['file_name'] = str(file)
                doc = open(file).read()
                for i, chunk_enc in enumerate(self.process_doc(doc)):
                    count += 1
                    self.status['chunk'] = i
                    self.status['total'] = count
                    yield chunk_enc
                    
    
    def process_doc(self, doc):
        """
        Return the tokenization result of chunks of one document.
        """
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