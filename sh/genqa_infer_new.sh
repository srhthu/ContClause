for step in 33444;
do
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--num_processes 2 --main_process_port 9014 -m cont_gen.infer_genqa_new \
--data_path data/cuad_clean/flan-t5_512/genqa/quest_test_max-asw-len_80.jsonl \
--base_model google/flan-t5-large \
--ckpt_dir runs/genqa/flan-t5-large_quest_lr1e-4_bs16_reformat_tgt512/checkpoint-${step} \
--save_path runs/genqa/flan-t5-large_quest_lr1e-4_bs16_reformat_tgt512/preds_ckpt_${step}.jsonl \
--dtype fp32 --batch_size 2
done