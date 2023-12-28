"""
An example to explore the iterable dataset.
"""
# %%
import torch
import math
import time
# %%
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))

# %%
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=23)

# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

# Mult-process loading with two worker processes
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# With even more workers
print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
# %%
class UnevenDataset(torch.utils.data.IterableDataset):
    """
    Each workloader return different number of examples.

    The i-th worker return range(i*10, i*10 + i), e.g., 2nd worker return [20,21]
    """
    def __init__(self):
        super().__init__()
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_worker = 1
            worker_id = 0
        else:
            num_worker = worker_info.num_workers
            worker_id = worker_info.id
        worker_id = worker_id + 1
        if worker_id > 9:
            return iter([])
        
        for i in range(worker_id*10, worker_id * 10 + worker_id):
            time.sleep(0.5)
            yield i
# %%
ds = UnevenDataset()
start = time.time()
for k in torch.utils.data.DataLoader(ds, num_workers = 3):
    cur_t = time.time() - start
    print(f'Time: {cur_t:.1f}s {k}')
# %%
class SimpleIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n
    
    def __iter__(self):
        for i in range(self.n):
            time.sleep(0.1)
            yield i
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
    
    def __getitem__(self, k):
        return k
    
    def __len__(self):
        return self.n
# %%
ds = SimpleIterDataset(10)
# ds = SimpleDataset(10)
# %%
for k in torch.utils.data.DataLoader(ds, num_workers = 2):
    print(k)
# %%
list(iter(torch.utils.data.DataLoader(ds, batch_size = 3)))
# %%
