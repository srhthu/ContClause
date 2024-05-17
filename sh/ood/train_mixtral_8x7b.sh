pmt_name=pmt_01 # pmt_01 pmt_01_yes_no
weight_decay=0.0
tbs=16
lr=1e-5

tk_name=mistral
base_model='mistralai/Mixtral-8x7B-v0.1'

for pmt_name in pmt_01; do
    for split_name in seed42_tr29; do
        HF_HUB_CACHE=/next_share/hf_cache/hub CUDA_VISIBLE_DEVICES=0,1,2,3 python \
        -m cont_gen.run.train_sft \
        --output_dir runs/ood/mixtral/${split_name}/${pmt_name}_all_lr${lr}_bs${tbs}_wd${weight_decay} \
        --data_path data/ood_split/${split_name}/${tk_name}/${pmt_name}/train_data.jsonl \
        --base_model ${base_model} --dtype bf16 --lora --lora_all_linear \
        --quantization `#--gradient_checkpointing` \
        --lr $lr --weight_decay ${weight_decay} --total_batch_size ${tbs} --device_batch_size 1 \
        --max_epochs 5 --logging_steps 10 --save_epochs 1
    done
done