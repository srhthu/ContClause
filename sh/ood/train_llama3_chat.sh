pmt_name=pmt_01 # pmt_01 pmt_01_yes_no
weight_decay=0.0
tbs=16
lr=1e-5

# tk_name=llama2
# base_model='meta-llama/Llama-2-7b-hf'

tk_name=llama3
base_model='meta-llama/Meta-Llama-3-8B-Instruct'

for pmt_name in pmt_01 pmt_01_yes_no; do
    for split_name in seed89_tr29 seed128_tr29; do
        HF_HUB_CACHE=/next_share/hf_cache/hub CUDA_VISIBLE_DEVICES=0,1 python \
        -m cont_gen.run.train_sft \
        --is_chat \
        --output_dir runs/ood/${tk_name}_chat/${split_name}/${pmt_name}_all_lr${lr}_bs${tbs}_wd${weight_decay} \
        --data_path data/ood_split/${split_name}/${tk_name}/${pmt_name}/train_data.jsonl \
        --base_model ${base_model} --dtype bf16 --lora --lora_all_linear \
        --lr $lr --weight_decay ${weight_decay} --total_batch_size ${tbs} --device_batch_size 1 \
        --max_epochs 5 --logging_steps 10 --save_epochs 1
    done
done