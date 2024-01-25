# infer by phi model
# run_dir_name=phi-15_quest_desc_lr1e-5
run_dir_name="phi-15_quest_lr1e-5_pretrain_1e-4_51200"


for ckpt in 25083; # 4180 8360 12540
do
CUDA_VISIBLE_DEVICES=1 python -m cont_gen.infer_genqa \
--prompt_path data/cuad_prompts/test_prompts_quest.jsonl \
--base_model /storage_fast/rhshui/llm/ms-phi-1_5 \
--save_path runs/genqa/${run_dir_name}/preds_ckpt_${ckpt}.jsonl \
--ckpt_dir runs/genqa/${run_dir_name}/checkpoint-${ckpt} \
--dtype bf16 --batch_size 8
done