base_model=/storage_fast/rhshui/llm/ms-phi-1_5
data_path=data/cuad_prompts/train_prompts_quest.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
--num_processes 4 --main_process_port 8989 -m cont_gen.train_genqa \
--output_dir runs/debug_genqa/phi-15_ds \
--ds_config ds_config/zero1_bf16.json \
--debug \
--data_path $data_path \
--max_length 1024 \
--base_model $base_model \
--dtype bf16 \
--lr 1e-5 \
--weight_decay 0.01 \
--device_batch_size 1 \
--grad_acc_steps 4 \
--max_epochs 1 \
--logging_steps 2 \
--save_epochs 1 \
--save_total_limit 3