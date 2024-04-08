# infer by phi model
# run_dir_name=phi-15_quest_desc_lr1e-5
# run_dir_name="phi-15_quest_lr1e-5_pretrain_1e-4_51200"
# run_dir_name="phi-15_quest_lr1e-5_pretrain_1e-5_307200"

# for ckpt in 8361 16722 25083; # 4180 8360 12540
# do
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
# --num_processes 2 --main_process_port 9013 -m cont_gen.infer_genqa \
# --prompt_path data/cuad_prompts/test_prompts_quest.jsonl \
# --base_model /storage_fast/rhshui/llm/ms-phi-1_5 \
# --save_path runs/genqa/${run_dir_name}/preds_ckpt_${ckpt}.jsonl \
# --ckpt_dir runs/genqa/${run_dir_name}/checkpoint-${ckpt} \
# --dtype bf16 --batch_size 2
# done

# Flan-T5
run_dir_name="flan-t5-large_quest_lr1e-4_bs16_reformat_tgt512"
for ckpt in 25083; # 4180 8360 12540
do
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--num_processes 2 --main_process_port 9013 -m cont_gen.infer_genqa \
--prompt_path data/cuad_prompts/test_prompts_quest.jsonl \
--base_model google/flan-t5-large \
--save_path runs/genqa/${run_dir_name}/preds_olddata_ckpt_${ckpt}.jsonl \
--ckpt_dir runs/genqa/${run_dir_name}/checkpoint-${ckpt} \
`#--ckpt_dir runs/genqa/flan-t5-large_quest_lr1e-4_bs16_new/checkpoint-25146` \
--dtype fp32 --batch_size 2
done