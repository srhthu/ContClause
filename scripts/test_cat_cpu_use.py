"""
We find that torch.cat use many cpus. check this issue.
"""
import time
import torch
from tqdm import tqdm
from accelerate import Accelerator
from typing import Mapping
import random
import psutil
from transformers import AutoModelForQuestionAnswering

def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")

def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)
        # return tensor2

    return None

def nested_to_cpu(tensors):
    "Transfer `tensors` to cpu (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_to_cpu(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_to_cpu(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t

def main():
    acc = Accelerator()

    print('Initialize tensors to cuda')
    tensors = [torch.rand((128,2048)).cuda(0) for _ in range(200)]
    tensors = [{
        'start_logits': torch.rand(16, 512).cuda(0), 
        'end_logits': torch.rand(16, 512).cuda(0)} 
            for _ in range(200)
    ]

    all_host = nested_to_cpu(tensors[0])

    for t in tqdm(tensors[1:]):
        t = nested_to_cpu(t)
        # all_host = torch.cat((all_host, t), dim = 0)
        all_host = nested_concat(all_host, t)
        time.sleep(0.15)

    for k,v in all_host.items():
        print(k, v.shape)

def main2():
    bs = 32
    data = [torch.randn(3, 128, 128) for _ in range(1000)] # Replay buffer

    N = 10
    total_time = 0
    total_usage = 0.
    for _ in range(N):
        s = random.sample(data, bs)
        t0 = time.time()
        x = torch.stack(s)
        total_time += time.time() - t0
        total_usage += psutil.cpu_percent(0.1)
        ... # Other things: Forward + Backward + Step

    print('Time per call:', 1000 * total_time / N, 'ms')
    print('Average usage:', total_usage / N, '%')

if __name__ == '__main__':
    main()
