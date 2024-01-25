run_dir_name=flan-t5-large_quest_desc_lr1e-4_bs16 # flan-t5-large_quest_lr1e-4_bs16 phi-15_quest_lr1e-5

for ckpt in 6270; # flan: 8361 16722 25083;
do
CUDA_VISIBLE_DEVICES=3 python -m cont_gen.infer_genqa \
--prompt_path data/cuad_prompts/test_prompts_quest.jsonl \
--base_model google/flan-t5-large \
--save_path runs/genqa/${run_dir_name}/preds_ckpt_${ckpt}.jsonl \
--ckpt_dir runs/genqa/${run_dir_name}/checkpoint-${ckpt} \
--dtype fp32 --batch_size 16 --max_length 900
done