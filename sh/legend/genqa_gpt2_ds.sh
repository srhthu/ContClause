base_model=gpt2
data_path=data/cuad_prompts/train_prompts_quest.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
--num_processes 4 --main_process_port 8989 -m cont_gen.train_genqa \
--output_dir runs/genqa/gpt2_quest_ds_2 \
--ds_config ds_config/zero1_fp32.json \
--data_path $data_path \
--max_length 1023 \
--base_model $base_model \
--dtype fp32 \
--lr 1e-5 \
--weight_decay 0.01 \
--device_batch_size 2 \
--grad_acc_steps 2 \
--max_epochs 1 \
--logging_steps 50 \
--save_epochs 1 \
--save_total_limit 3