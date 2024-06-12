split=seed42_tr29 # seed42_tr29 seed89_tr29 seed128_tr29
tk_name=flan-t5
pmt_name=pmt_01_yes_no # pmt_01, pmt_01_yes_no
ood=ood

model_name=t5-large # flan-t5-large t5-large flan-t5-xl

export HF_HUB_CACHE=/next_share/hf_cache/hub

for pmt_name in pmt_01; do
    for split in seed42_tr29 seed89_tr29 seed128_tr29; do
        for ood in ood id; do
            CUDA_VISIBLE_DEVICES=2 python -m cont_gen.run.infer_sft \
            --is_seq2seq \
            --data_path data/ood_split/${split}/flan-t5/${pmt_name}/test_data_${ood}.jsonl \
            --part sampled \
            --run_dir runs/ood/${model_name}/${split}/${pmt_name}_lr2e-4_bs16_wd0.0/ \
            --dtype bf16 --batch_size 1 
        done
    done
done