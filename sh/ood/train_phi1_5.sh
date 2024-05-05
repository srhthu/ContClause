# Do not use DDP
split_name=seed42_tr29
pmt_name=pmt_01_yes_no # pmt_01
weight_decay=0.0
tbs=16
lr=1e-5

base_model=/storage_fast/rhshui/llm/ms-phi-1_5

# CUDA_VISIBLE_DEVICES=6,7 python \
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 8923 \
-m cont_gen.run.train_sft \
--output_dir runs/ood/phi-1_5/${split_name}/${pmt_name}_lr${lr}_bs${tbs}_wd${weight_decay} \
--data_path data/ood_split/${split_name}/${pmt_name}/train_data.jsonl \
--max_length 512 --base_model ${base_model} --dtype bf16 \
--lr $lr --weight_decay ${weight_decay} --total_batch_size ${tbs} --device_batch_size 1 \
--max_epochs 5 --logging_steps 10 --save_epochs 1