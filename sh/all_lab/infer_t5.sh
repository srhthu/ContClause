split=seed42_tr29 # seed42_tr29 seed89_tr29 seed128_tr29
pmt_name=pmt_01 
ood=ood

model_name=flan-t5-xl # flan-t5-large t5-large flan-t5-xl

export HF_HUB_CACHE=/next_share/hf_cache/hub


for split in seed128_tr29; do
    for ood in ood; do
        CUDA_VISIBLE_DEVICES=2 python -m cont_gen.run.infer_sft \
        --is_seq2seq \
        --data_path data/ood_split/${split}/flan-t5/${pmt_name}/test_data_${ood}.jsonl \
        --part sampled \
        --run_dir runs/all_lab/${model_name}/${pmt_name}_lr1e-4_bs16_wd0.0/ \
        --save_prefix "${split}/" \
        --dtype bf16 --batch_size 1 
    done
done
