"""
Explore iterable dataset in distributed training.
"""
import time
import logging
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
import accelerate
from transformers import AutoTokenizer
import numpy as np

from clm.data_loader import CUAD_FileLoader_Clean

logging.basicConfig(level = 20)

class MyIterDataset(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n
        state = accelerate.PartialState()
        self.local_rank = state.process_index
    
    def __iter__(self):
        for i in range(self.n):
            time.sleep(0.2)
            logging.info(f'Handle data {i} by Process {self.local_rank}')
            yield i

class ListIterDataset(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n
    
    def __iter__(self):
        for i in range(self.n):
            yield (i*100+1, i*100+2, i*100 + 3)

class DictIterDataset(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n
    
    def __iter__(self):
        for i in range(self.n):
            yield {'x': np.array([i, -i]), 'y': i*100}

def main():

    accelerator = accelerate.Accelerator()
    local_rank = accelerator.process_index

    tk = AutoTokenizer.from_pretrained('gpt2')
    tk.add_special_tokens({'pad_token': '[PAD]'})
    ds = CUAD_FileLoader_Clean(
        '../data/cuad_contracts_small', 
        tokenizer = tk, max_length = 1024, seed=42
    )

    # ds = MyIterDataset(21)
    # ds = ListIterDataset(9)
    # ds = DictIterDataset(9)

    dl = DataLoader(ds, batch_size = 2)
    print(type(dl))
    # exit()


    dl = accelerator.prepare(dl)

    # for batch in dl:
    #     logging.info(f'Process {local_rank} : {batch}')
    
    # logging.info(f'New iteration')
    # for batch in dl:
    #     logging.info(f'Process {local_rank} : {batch}')
    
    if local_rank == 0:
        print(dl.__class__.__name__)
    for i, batch in enumerate(dl):
        if local_rank == 0:
            print(f'Step {i+1} : {ds.status["index"]}', flush = True)
        dist.barrier()
    if local_rank == 0:
        print(f'New iteration')
    for i, batch in enumerate(dl):
        if local_rank == 0:
            print(f'Step {i+1} : {ds.status["index"]}', flush = True)
        dist.barrier()

if __name__ == '__main__':
    main()