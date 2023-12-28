import torch
from accelerate import Accelerator

def main():
    acc = Accelerator()
    x = torch.tensor(1., device = acc.device)
    gx = acc.gather(x)
    if acc.is_main_process:
        print(gx.shape)
        print(gx.mean())

if __name__ == '__main__':
    main()