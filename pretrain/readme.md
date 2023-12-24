# Pretraining Causal LM
```
Note: all codes are run under this folder
```

## Find Device_Batch_Size
```Bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 5677 find_max_batch.py --model_path gpt2 --dtype fp32 --max_length 512 --max_device_batch_size 8
```