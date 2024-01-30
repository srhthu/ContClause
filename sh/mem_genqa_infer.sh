CUDA_VISIBLE_DEVICES=2 python -m cont_gen.infer_mem_genqa \
--test_path data/cuad_clean/flan-t5_512/test_paras.jsonl \
--base_model google/flan-t5-large \
--model_ckpt runs/mem_genqa/flan-t5-large_quest_lr1e-4_bs16/checkpoint-12492 \
--save_path runs/mem_genqa/flan-t5-large_quest_lr1e-4_bs16/preds_ckpt_12492.jsonl \
--quests data/clause/prompt_quest.json \
--dtype fp32 --batch_size 6