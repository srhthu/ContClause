model_size=large # base, large, xl, xxl
data_suffix=quest # quest, quest_desc

base_model=google/flan-t5-${model_size}
data_path=data/cuad_clean/flan-t5_512/memory_genqa_${data_suffix}_train.jsonl
lr=1e-4

CUDA_VISIBLE_DEVICES=1 accelerate launch \
--num_processes 1 --main_process_port 8922 -m cont_gen.train_mem_genqa \
--output_dir runs/mem_genqa/flan-t5-${model_size}_${data_suffix}_lr${lr}_bs16 \
`#--ds_config ds_config/zero1_bf16.json` \
--data_path $data_path \
--max_length 900 --max_target_length 256 \
--base_model $base_model \
--dtype bf16 \
--lr $lr \
--weight_decay 0.01 \
--device_batch_size 1 \
--grad_acc_steps 16 \
--max_epochs 5 \
--logging_steps 10 \
--save_epochs 1 \
--save_total_limit 3