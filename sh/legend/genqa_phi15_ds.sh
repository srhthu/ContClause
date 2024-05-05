data_suffix=quest # quest, quest_desc
base_model=/storage_fast/rhshui/llm/ms-phi-1_5
data_path=data/cuad_prompts/train_prompts_${data_suffix}.jsonl
lr=1e-5
saved_model='pretrain/runs/cuad_phi-15_zero1_lr1e-5/model_state/step-307200/pytorch_model.bin' # ''
model_suffix='_pretrain_1e-5_307200' # ""

CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
--num_processes 2 --main_process_port 8989 -m cont_gen.train_genqa \
--output_dir runs/genqa/phi-15_${data_suffix}_lr${lr}${model_suffix} `#--debug` \
`#--ds_config ds_config/zero1_bf16.json` \
--data_path $data_path \
--max_length 1100 \
--base_model $base_model \
--saved_model $saved_model \
--dtype bf16 \
--lr $lr \
--weight_decay 0.01 \
--device_batch_size 1 \
--grad_acc_steps 8 \
--max_epochs 3 \
--logging_steps 10 \
--save_epochs 1 \
--save_total_limit 3