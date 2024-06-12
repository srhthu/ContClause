tk_name=flan-t5
pmt_name=pmt_01 # pmt_01, pmt_01_yes_no
model_name=flan-t5-xl # flan-t5-large t5-large flan-t5-xl
arg_seq='--is_seq2seq'
model_path=google/flan-t5-xl

# tk_name=llama3
# pmt_name=pmt_01 # pmt_01, pmt_01_yes_no
# model_name=llama3
# arg_seq='--is_chat'
# model_path=meta-llama/Meta-Llama-3-8B-Instruct

export HF_HUB_CACHE=/next_share/hf_cache/hub

for split in seed89_tr29 seed128_tr29; do
    for ood in ood; do
        CUDA_VISIBLE_DEVICES=1 python -m cont_gen.run.infer_sft \
        ${arg_seq} \
        --data_path data/ood_split/${split}/${tk_name}/${pmt_name}/test_data_${ood}.jsonl \
        --part sampled \
        --base_model ${model_path} \
        --ckpt_dir ${model_path} \
        --save_path runs/baseline/${model_name}/${split}/${pmt_name}/predictions_${ood}_sampled.jsonl \
        --dtype bf16 --batch_size 1 
    done
done