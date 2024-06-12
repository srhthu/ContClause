pmt_name=pmt_01
weight_decay=0.0
tbs=16
lr=1e-5

# tk_name=llama3
# base_model='meta-llama/Meta-Llama-3-8B'
# run_name=llama3
# chat_args=''

# tk_name=llama3
# base_model='meta-llama/Meta-Llama-3-8B-Instruct'
# run_name=llama3_chat
# chat_args='--is_chat'

# tk_name=mistral
# base_model='mistralai/Mistral-7B-v0.1'
# run_name=mistral
# chat_args=''

tk_name=mistral
base_model='mistralai/Mistral-7B-Instruct-v0.2'
run_name=mistral_chat
chat_args='--is_chat'

HF_HUB_CACHE=/next_share/hf_cache/hub CUDA_VISIBLE_DEVICES=2,3 python -m cont_gen.run.train_sft \
    ${chat_args} \
    --data_path data/cuad_sft/${tk_name}/${pmt_name}/train_data.jsonl \
    --output_dir runs/all_lab/${run_name}/${pmt_name}_lr${lr}_bs${tbs}_wd${weight_decay} \
    --base_model ${base_model} --dtype bf16 --lora --lora_all_linear \
    --lr $lr --weight_decay ${weight_decay} --total_batch_size ${tbs} --device_batch_size 1 \
    --max_epochs 5 --logging_steps 10 --save_epochs 1
