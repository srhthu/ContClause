model_size=large # base, large, xl, xxl
data_suffix=quest # quest, quest_desc, desc_quest

base_model=google/flan-t5-${model_size}
# data_path=data/cuad_clean/flan-t5_512/genqa/${data_suffix}_train_max-asw-len_80_fix.jsonl
data_path=data/cuad_prompts/train_prompts_src_quest.jsonl

lr=1e-4

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
--num_processes 2 --main_process_port 8922 -m cont_gen.train_mem_genqa \
--output_dir runs/genqa/flan-t5-${model_size}_${data_suffix}_lr${lr}_bs16_reformat_tgt512 \
`#--ds_config ds_config/zero1_bf16.json` \
--data_path $data_path \
--max_length 900 --max_target_length 512 \
--base_model $base_model \
--dtype bf16 \
--lr $lr \
--weight_decay 0.01 \
--device_batch_size 1 \
--grad_acc_steps 8 \
--max_epochs 5 \
--logging_steps 10 \
--save_epochs 1 \
--save_total_limit 5