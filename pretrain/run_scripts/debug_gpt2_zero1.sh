# set max_epoch to 10
# step_arg='--max_epochs 10'
# ckpt_arg=''
# run_sufix='_epoch'

# resume from epoch 10 and train epoch 12
step_arg='--max_epochs 12'
ckpt_arg='--ckpt_dir runs/debug_gpt2_zero1_epoch/checkpoint-800'
run_sufix='_epoch'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 5812 train_clm.py \
--output_dir runs/debug_gpt2_zero1${run_sufix} \
$ckpt_arg \
--data_dir ../data/cuad_contracts_small \
--model_path gpt2 \
--dtype bf16 \
--lr 1e-4 \
--device_bs 2 \
--grad_acc_steps 4 \
$step_arg \
--log_steps 5 \
--save_steps 100 \
--save_total_limit 3 \
--max_length 1024 \
--ds_config ds_config/zero1_bf16.json
