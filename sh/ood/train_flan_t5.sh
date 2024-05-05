model_size=large # base, large, xl, xxl
split_name=seed42_tr29
pmt_name=pmt_01_yes_no # pmt_01
weight_decay=0.0
tbs=16
lr=1e-4

base_model=google/flan-t5-${model_size}
data_path=data/ood_split/${split_name}/${pmt_name}/train_data.jsonl

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 8922 \
-m cont_gen.run.train_sft \
--output_dir runs/ood/flan-t5-${model_size}/${split_name}/${pmt_name}_lr${lr}_bs${tbs}_wd${weight_decay} \
--data_path ${data_path} \
--max_length 512 --base_model ${base_model} --dtype bf16 \
--lr $lr --weight_decay ${weight_decay} --total_batch_size ${tbs} --device_batch_size 2 \
--max_epochs 5 --logging_steps 10 --save_epochs 1