split=seed42_tr29 # seed42_tr29 seed89_tr29 seed128_tr29

pmt_name=pmt_01 # pmt_01, pmt_01_yes_no
ood=id

# tk_name=llama3
# mod_name=llama3
# chat_args=''

# tk_name=llama3
# mod_name=llama3_chat
# chat_args='--is_chat'

# tk_name=mistral
# mod_name=mistral
# chat_args=''

tk_name=mistral
mod_name=mistral_chat
chat_args='--is_chat'

export HF_HUB_CACHE=/next_share/hf_cache/hub

for split in seed42_tr29 seed89_tr29 seed128_tr29; do
    for ood in ood; do
        CUDA_VISIBLE_DEVICES=2  python -m cont_gen.run.infer_sft \
        ${chat_args} \
        --data_path data/ood_split/${split}/${tk_name}/${pmt_name}/test_data_${ood}.jsonl \
        --part sampled \
        --run_dir runs/all_lab/${mod_name}/${pmt_name}_lr1e-5_bs16_wd0.0/ \
        --save_prefix "${split}/" \
        --dtype bf16 --batch_size 1 
    done
done