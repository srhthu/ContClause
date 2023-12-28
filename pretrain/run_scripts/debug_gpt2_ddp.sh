# Comment --output_dir to check no output case
# add ckpt_dir to check load from ckpt
# check max_steps and max_epochs works well

# # first run 200 steps
# step_arg='--max_steps 200'
# ckpt_arg=''

# then load checkpoint-200 and run up to 800 steps, check the effect of save_total_limit
# step_arg='--max_steps 800'
# ckpt_arg='--ckpt_dir runs/debug_gpt2_ddp/checkpoint-200'

# set max_epoch to 10
step_arg='--max_epochs 10'
ckpt_arg=''
run_sufix='_epoch'

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 5812 train_clm.py \
--output_dir runs/debug_gpt2_ddp${run_sufix} \
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
--max_length 1024
