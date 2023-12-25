# Pretraining Causal LM
```
Note: all codes are run under this folder
```

## Find Device_Batch_Size
```Bash
# DDP
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 5677 find_max_batch.py --model_path microsoft/phi-1_5 --dtype bf16 --max_length 1024 --max_device_batch_size 8

# Deepspeed Zero 1
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 5677 find_max_batch.py --model_path microsoft/phi-1_5 --dtype bf16 --max_length 1024 --max_device_batch_size 8 --ds_config ds_config/zero1_bf16.json
```