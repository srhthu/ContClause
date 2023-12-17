"""
Explore iterable dataset in distributed training.
"""
import time
from torch.utils.data import IterableDataset, DataLoader
import accelerate

import logging

logging.basicConfig(filename = './log.txt', level = 10)

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

def main():
    accelerator = accelerate.Accelerator()
    
    ds = MyIterDataset(20)
    dl = DataLoader(ds, batch_size = 2)

    dl = accelerator.prepare(dl)

    local_rank = accelerator.process_index

    for batch in dl:
        logging.info(f'Process {local_rank} : {batch}')

if __name__ == '__main__':
    main()