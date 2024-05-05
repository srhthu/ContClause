for ckpt in 3721 7442 11163 14884 18605;
do
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 9013 \
-m cont_gen.run.infer_sft \
--data_path data/ood_split/seed42_tr29/pmt_01_yes_no/test_data_ood.jsonl \
--part sampled \
--base_model google/flan-t5-large \
--ckpt_dir runs/ood/flan-t5-large/seed42_tr29/pmt_01_yes_no_lr1e-4_bs16_wd0.0/checkpoint-${ckpt} \
--dtype fp32 --batch_size 4 
done