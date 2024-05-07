split=seed42_tr29
tk_name=llama3
pmt_name=pmt_01_yes_no
ood=ood

for ckpt in 15692 31384 47076 62768 78460;
do
CUDA_VISIBLE_DEVICES=0 `#accelerate launch --num_processes 2 --main_process_port 9013` python \
-m cont_gen.run.infer_sft \
--data_path data/ood_split/${split}/${llama3}/${pmt_name}/test_data_${ood}.jsonl \
--part all \
--base_model meta-llama/Meta-Llama-3-8B \
--ckpt_dir runs/ood/${tk_name}/${split}/${pmt_name}_all_lr1e-5_bs16_wd0.0/checkpoint-${ckpt} \
--dtype bf16 --batch_size 1 
done