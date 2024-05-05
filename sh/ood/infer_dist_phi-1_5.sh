for ckpt in 29772 # 14886 29772 44658 59544 74430;
do
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 9014 \
-m cont_gen.run.infer_sft \
--data_path data/ood_split/seed42_tr29/pmt_01_yes_no/test_data_ood.jsonl \
--part sampled \
--base_model /storage_fast/rhshui/llm/ms-phi-1_5 \
--ckpt_dir runs/ood/phi-1_5/seed42_tr29/pmt_01_yes_no_lr1e-5_bs16_wd0.0/checkpoint-${ckpt} \
--dtype fp32 --batch_size 4 
done