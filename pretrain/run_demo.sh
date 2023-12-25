# model_path=/storage_fast/rhshui/llm/llama2_hf/llama-2-7b
# model_path=google/flan-t5-large
model_path=gpt2
# model_path=microsoft/phi-1_5

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master-port 5567 -m pretrain.demo_train_ds \
# --model_path $model_path --dtype bf16 \
# --max_steps 200 --device_bs 2 --grad_acc_steps 4 --log_steps 5 \
# --lib deepspeed --ds_config pretrain/ds_config/zero2_bf16.json

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -m pretrain.demo_train_ds --model_path $model_path --dtype bf16

for device_bs in 2; do
for gaccstep in 4;do
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 5677 demo_train_ds.py \
--model_path $model_path --dtype bf16 \
--max_steps 200 --device_bs ${device_bs} --grad_acc_steps ${gaccstep} --log_steps 5 \
--lib accelerate --ds_config ./ds_config/zero1_fp16.json --lr 1e-5
done
done

# CUDA_VISIBLE_DEVICES=1,2 deepspeed --num_gpus 2 --master_port 5689 --module pretrain.demo_train_ds \
# --model_path $model_path --dtype bf16 \
# --max_steps 200 --device_bs 2 --grad_acc_steps 4 --log_steps 5 \
# --lib deepspeed --ds_config pretrain/ds_config/zero2_bf16.json
