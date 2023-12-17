"""
Explore iterable dataset in distributed training.
"""
import time
from torch.utils.data import IterableDataset, DataLoader
import accelerate


class MyIterDataset(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n
    
    def __iter__(self):
        for i in range(self.n):
            time.sleep(0.2)
            yield i

def main():
    accelerator = accelerate.Accelerator()
    
    ds = MyIterDataset(50)
    dl = DataLoader(ds, batch_size = 2)

    dl = accelerator.prepare(dl)

    local_rank = accelerator.process_index

    for batch in dl:
        print(f'{local_rank} : {batch}')

if __name__ == '__main__':
    main()