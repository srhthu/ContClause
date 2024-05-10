split=seed42_tr29 # seed42_tr29 seed89_tr29 seed128_tr29
tk_name=llama3
pmt_name=pmt_01 # pmt_01, pmt_01_yes_no
ood=ood
ckpt=15692 #15692 31384 47076 62768 78460

export HF_HUB_CACHE=/next_share/hf_cache/hub

# for ckpt in 15692 31384 47076 62768 78460;
# do
CUDA_VISIBLE_DEVICES=3 `#accelerate launch --num_processes 2 --main_process_port 9013` python \
-m cont_gen.run.infer_sft \
--is_chat \
--data_path data/ood_split/${split}/${tk_name}/${pmt_name}/test_data_${ood}.jsonl \
--part sampled \
`#--ckpt_dir runs/ood/${tk_name}/${split}/${pmt_name}_all_lr1e-5_bs16_wd0.0/checkpoint-${ckpt}` \
--run_dir runs/ood/${tk_name}_chat/${split}/${pmt_name}_all_lr1e-5_bs16_wd0.0/ \
--dtype bf16 --batch_size 1 
# done