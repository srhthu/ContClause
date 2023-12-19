"""
A demo of pretraining using deepspeed.

Aim:
    providing a toy example
    investigating memory usage of different input seq length.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--dtype', choices = ['fp16', 'bf16'], default = 'fp16')
    # parser.add_argument()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code = True)

    torch_dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch_dtype_map[args.dtype],
        device_map = 'auto'
    )
    print(model.hf_device_map)

if __name__ == '__main__':
    main()