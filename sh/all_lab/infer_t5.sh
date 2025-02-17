split=seed42_tr29 # seed42_tr29 seed89_tr29 seed128_tr29
pmt_name=pmt_01 
ood=ood

model_name=flan-t5-xl # flan-t5-large t5-large flan-t5-xl
part=sampled
export HF_HUB_CACHE=/next_share/hf_cache/hub


CUDA_VISIBLE_DEVICES=2 python -m cont_gen.run.infer_sft \
    --is_seq2seq \
    --data_path data/cuad_sft/flan-t5/${pmt_name}/test_data.jsonl \
    --part ${part} \
    --run_dir runs/all_lab/${model_name}/${pmt_name}_lr1e-4_bs16_wd0.0/ \
    --save_path runs/all_lab/${model_name}/${pmt_name}_lr1e-4_bs16_wd0.0/predictions_${part}.jsonl \
    --dtype bf16 --batch_size 1 

