step_arg='--max_epochs 1'
# ckpt_arg='--ckpt_dir runs/debug_gpt2_zero1_epoch/checkpoint-800'

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 5812 train_clm.py \
--output_dir runs/cuad_phi-15_zero1_lr1e-5 \
$ckpt_arg \
--data_dir ../data/cuad_contracts \
--model_path microsoft/phi-1_5 \
--dtype bf16 \
--lr 1e-4 \
--device_bs 1 \
--grad_acc_steps 32 \
$step_arg \
--log_steps 4 \
--save_steps 51200 \
--save_total_limit 3 \
--max_length 1024 \
--ds_config ds_config/zero1_bf16.json
