model_size=large # base, large, xl, xxl
data_suffix=quest # quest, quest_desc

base_model=google/flan-t5-${model_size}
data_path=data/cuad_prompts/train_prompts_${data_suffix}.jsonl
lr=1e-4

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
--num_processes 2 --main_process_port 8991 -m cont_gen.train_genqa \
--output_dir runs/genqa/flan-t5-${model_size}_${data_suffix}_lr${lr}_bs16_re1 \
`#--ds_config ds_config/zero1_bf16.json` \
--data_path $data_path \
--max_length 512 \
--base_model $base_model \
--dtype fp32 \
--lr $lr \
--weight_decay 0.01 \
--device_batch_size 1 \
--grad_acc_steps 8 \
--max_epochs 3 \
--logging_steps 10 \
--save_epochs 1 \
--save_total_limit 3