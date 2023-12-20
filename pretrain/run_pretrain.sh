model_path=/storage_fast/rhshui/llm/llama2_hf/llama-2-7b
model_path=google/flan-t5-large
model_path=gpt2

# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc-per-node 2 --master-port 5567 -m pretrain.demo_train_ds \
# --model_path $model_path --dtype bf16 \
# --ds_config pretrain/ds_config/ds_stage2.json 

CUDA_VISIBLE_DEVICES=3 python -m pretrain.demo_train_ds --model_path $model_path 