model_size=xl # base, large, xl, xxl
pmt_name=pmt_01 
weight_decay=0.0
tbs=16
lr=1e-4

base_model=google/flan-t5-${model_size}
mod_name=flan-t5-${model_size}


HF_HUB_CACHE=/next_share/hf_cache/hub CUDA_VISIBLE_DEVICES=0,1 python -m cont_gen.run.train_sft \
    --data_path data/cuad_sft/flan-t5/${pmt_name}/train_data.jsonl \
    --output_dir runs/all_lab/${mod_name}/${pmt_name}_lr${lr}_bs${tbs}_wd${weight_decay} \
    --base_model ${base_model} --dtype bf16 \
    --lr $lr --weight_decay ${weight_decay} --total_batch_size ${tbs} --device_batch_size 1 \
    --max_epochs 5 --logging_steps 10 --save_epochs 1