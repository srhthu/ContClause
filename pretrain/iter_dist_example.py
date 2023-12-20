"""
Explore iterable dataset in distributed training.
"""
import time
import logging
from torch.utils.data import IterableDataset, DataLoader
import accelerate
from transformers import AutoTokenizer
import numpy as np

from pretrain.data_loader import FileLoader

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
    
    # ds = MyIterDataset(20)

    # tk = AutoTokenizer.from_pretrained('google/flan-t5-large')
    # tk.add_special_tokens({'pad_token': '[PAD]'})
    # ds = FileLoader(
    #     '/storage_fast/rhshui/workspace/datasets/legal/cuad_contracts', 
    #     tokenizer = tk, max_length = 52
    # )

    # ds = ListIterDataset(9)

    ds = DictIterDataset(9)

    dl = DataLoader(ds, batch_size = 2)
    print(type(dl))
    # exit()


    dl = accelerator.prepare(dl)

    for batch in dl:
        logging.info(f'Process {local_rank} : {batch}')

if __name__ == '__main__':
    main()