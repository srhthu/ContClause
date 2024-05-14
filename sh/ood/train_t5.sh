model_size=large # base, large, xl, xxl
split_name=seed42_tr29
pmt_name=pmt_01 # pmt_01 pmt_01_yes_no
weight_decay=0.0
tbs=16
lr=1e-4

base_model=google/t5-v1_1-${model_size}
mod_name=t5-${model_size}

# base_model=google/flan-t5-${model_size}
# mod_name=flan-t5-${model_size}

for split_name in seed42_tr29 seed89_tr29 seed128_tr29; do
    HF_HUB_CACHE=/next_share/hf_cache/hub CUDA_VISIBLE_DEVICES=3 python -m cont_gen.run.train_sft \
    --output_dir runs/ood/${mod_name}/${split_name}/${pmt_name}_lr${lr}_bs${tbs}_wd${weight_decay} \
    --data_path data/ood_split/${split_name}/flan-t5/${pmt_name}/train_data.jsonl \
    --base_model ${base_model} --dtype bf16 \
    --lr $lr --weight_decay ${weight_decay} --total_batch_size ${tbs} --device_batch_size 2 \
    --max_epochs 5 --logging_steps 10 --save_epochs 1
done