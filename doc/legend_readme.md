# Pipeline for Generative with Notes
We include some notes as extra context, including:
- Head of previous paragraphs
- Summary of previous paragraphs' clause types.

## Data Process

### Training Prompt Data
To get the source and target data for fine-tuning, run the script `cont_gen/data_process/build_genqa_with_context.py`. The input data is supposed to be paragraph data that has been partitioned to fit max token length.

Some configurable variables are:
- **max_mem_num**: max number of memory paragraphs
- **total_mem_len**: max number of tokens in the memory
- **max_answer_len**: max number of tokens in one piece of answer. The whole answer may contain many pieces. If the answer exceed the token limit, replace middle tokens with "..."
- **neg_ratio**: the ratio of negative samples to positive samples

Run the scripts:
```Bash
python -m cont_gen.data_process.build_genqa_with_context \
data/cuad_clean/flan-t5_512/CUAD_paras.jsonl \
data/cuad_clean/flan-t5_512/memory_genqa_quest_train.jsonl \
google/flan-t5-large \
data/clause/prompt_quest.json \
--split_titles data/cuad_split/ori_train_titles.json \
--max_mem_num 10 --total_mem_len 256 --max_answer_len 60 --neg_ratio 1.0
```

Output: 10330 positive 10330 negtive samples for train and oracle test.  
Statistics of max length: 900

## Inference
For each document, infer paragraphs one by one as previous clause information is need. (`cont_gen.infer_mem_genqa.py`)

**Run**
```Bash
python -m cont_gen.infer_mem_genqa \
--test_path data/cuad_clean/flan-t5_512/test_paras.jsonl \
--base_model google/flan-t5-large \
--model_ckpt runs/mem_genqa/flan-t5-large_quest_lr1e-4_bs16/checkpoint-12492 \
--save_path runs/mem_genqa/flan-t5-large_quest_lr1e-4_bs16/preds_ckpt_12492.jsonl \
--quests data/clause/prompt_quest.json \
--dtype fp32 --batch_size 6
```

# Pipeline for Generative Methods
## Overview

Before building feature, we prepare the questions for each clause. [Link](#question-prepare)

Further split paragraphs into chunks to fit in a max token length of a tokenizer. [Link](#split-by-tokenized-length)

Next, we build the full prompts with provisions (contract text), questions and answers. [Link]()

Next, we build features for training and test. [Link](#build-features)
- For training, we sample negative paragraphs for each question. 
- For test, we ask all question for each paragraph with length > 10

### Statistics
There are 510 contracts, 6702 available clauses annotated. Among them, 2594 clauses have multiple spans. 

Train prompts: 16,723
Test prompts: 229,518 (5598 chunks)

## Data Process (New)
```Bash
python -m cont_gen.data_process.build_qagen_new \
data/cuad_clean/flan-t5_512/CUAD_paras.jsonl \
data/cuad_clean/flan-t5_512/genqa/quest_train_max-asw-len_80.jsonl \
google/flan-t5-large \
--ommit_len 80 \
--quests data/clause/prompt_quest.json \
--split_titles data/cuad_split/ori_train_titles.json
```
Output: dict of **title, para_idx, q_id, source, target**

## ~~Data  Process~~

### Question Prepare
Prepare the questions for each clause. 

**Script**: `scripts/prepare_questions.py`

**Output**: `data/clause/`
- `clause_info.json`: a list of clause information extracted from readme file.
  Include "category", "desc", "answer_format" and "group"
- `prompt_quest_desc.json` and `prompt_quest.json` the prompt for questions. List[str]

### Split by Tokenized Length
Split paragraphs into chunks to fit in a max token length. Output is a flatten structure.

**Script**: `cont_gen/data_process/split_para_to_chunks.py`

**Run**
```Bash
python -m cont_gen.data_process.split_para_to_chunks data/cuad_clean/CUADv1_paras_merge.jsonl data/cuad_clean/CUADv1_chunks_merge.jsonl microsoft/phi-1_5 512
```

**Input**: paragraph data (merged): `data/cuad_clean/CUADv1_paras_merge.jsonl`

**Output**: 
`data/cuad_clean/CUADv1_chunks_merge.jsonl` in a flat structure.
```
title, chunk_index, text, para_idx, para_offset, qas
```

### Build SFT Data
Build full prompts. For training, we sample negative examples.

**Code**: `cont_gen/data_process/build_qagen_sft.py`

**Run**
```Bash
# suffix can be quest or quest_desc
suffix=quest
# suffix=quest_desc

# build training data
python -m cont_gen.data_process.build_qagen_sft \
data/cuad_clean/CUADv1_chunks_merge.jsonl \
data/cuad_prompts/train_prompts_${suffix}.jsonl \
microsoft/phi-1_5 80 \
--quest data/clause/prompt_${suffix}.json \
--train_split data/cuad_split/train_separate_questions.json

# build test data
python -m cont_gen.data_process.build_qagen_sft \
data/cuad_clean/CUADv1_chunks_merge.jsonl \
data/cuad_prompts/test_prompts_${suffix}.jsonl \
microsoft/phi-1_5 80 \
--quest data/clause/prompt_${suffix}.json \
--test_split data/cuad_split/test.json
```

**Output**: 
```
title
chunk_index
q_id
prompt: for training, contain answer. for test, no answer.
```

## Training
training code: `cont_gen/train_genqa.py`

Run the script: 
- `sh/genqa_phi15_ds.sh`: use deepspeed zero stage 1 to train the model

### Compute loss only on outputs
The training data should contain
```
input_ids: (batch_size, seq_len)
attention_mask: (batch_size, seq_len)
# source_len: (batch_size) length of the source part in prompt
labels: (batch_size, seq_len) # with pad token label to be -100
```

## Evaluating
### Get Predictions
```Bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--num_processes 2 --main_process_port 9023 \
-m cont_gen.infer_genqa \
--prompt_path data/cuad_prompts/test_prompts_quest.jsonl \
--base_model google/flan-t5-large \
--save_path runs/genqa/flan-t5-large_quest_lr1e-4_bs16/preds_ckpt_25083.jsonl \
--ckpt_dir runs/genqa/flan-t5-large_quest_lr1e-4_bs16/checkpoint-25083 \
--dtype fp32 --batch_size 16
```
**Output**: each sample is a dict of
```
title, chunk_index, q_id, prediction
```
### Get metrics
Macro F1 and IOU by token level overlapping:
```Bash
python -m cont_gen.evaluate.eval_genqa_result data/cuad_clean/CUADv1_chunks_merge.jsonl runs/genqa/flan-t5-large_quest_lr1e-4_bs16/preds_ckpt_8361.jsonl
```

# Pipeline for QA Span baseline
### Document Tokenization
Run `cont_gen/data_process/cut_doc.py` with argumetns:
```Bash
source .env
python -m cont_gen.data_process.cut_doc "${CUAD_PATH}/CUAD_v1.json" \
./data/doc/doc_tokens_roberta_rm.pkl \
roberta-base \
--remove_space --n_cpu 10
```
Output is a list of dict to store tokenized information:
- doc_id (`str`): the title of the document
- doc_token_ids (`List[int]`): token ids of the tokenizer
- token_to_char (`List[Tuple[int, int]]`): record the char span of every token. (End not included) 

### Build Features for QA Model
1 Contract -> 41 Example -> x positive features and x negtive features.

Each feature comprise a segment (window), a question and the answer (span position).

Run 
```Bash
source .env
python -m cont_gen.data_process.build_qa_feature \
--data_file ${CUAD_TRAIN} \
--doc_tk_path ./data/doc/doc_tokens_roberta_rm.pkl \
--output_path ./data/features/qa_roberta_train.pkl \
--tokenizer_path roberta-base --balance
```
Remove argument `--balance` when building test features.

Output is a list of features:
```
    input_ids
    attention_mask
    token_type_ids
    seq_len: non-pad token number
    paragraph_len: context of the document
    context_offset: start position of context in the input_ids
    span_start: answer span start token position in the document
    p_mask: 1 for tokens cannot be answers
    cls_index: used for impossible span
    start_position: answer span start token position in the input_ids
    end_position: answer span end token position in the input_ids
    is_impossible: whether the window has answer span
    qas_id: tuple of (doc_id, clause_id)
    example_index
```

### Train QA model with features (No evaluation)
```Bash
python -m cont_gen.train_qa --features data/features/qa_roberta_train.pkl --base_model roberta-base --output_dir runs/qa/roberta-base_lr1e-4_bs16 --num_epoch 5 --lr 1e-4 --batch_size 16
```
Output:  each result is a dict of 'start_logits', 'end_logits'

### Infer QA model
For each feature, output `start_logits` and `end_logits`
```Bash
# for debug
python -m cont_gen.infer_qa --save_path runs/debug/model_outputs.pkl --ckpt roberta-base --features data/features/qa_roberta_test.pkl

# infer one checkpoint
python -m cont_gen.infer_qa --features data/features/qa_roberta_test.pkl --ckpt runs/qa/roberta-base_lr1e-4_bs16/checkpoint-13000

# infer all checkpoints
python -m cont_gen.infer_qa --features data/features/qa_roberta_test.pkl --exp_dir  runs/qa/roberta-base_lr1e-4_bs16
```

### Compute Predictions
contract -> examples -> features -> start_logits, end_logits

For each feature, propose n_best start and end indexes, and save all possible combinations as preliminary predictions:
```
    start_index
    end_index
    start_logit
    end_logit
```
Get the prediction of each QA example (`qa_id`). Given all prelim predictions, filter top ranking predictions, and get the answer span text and position
```Json
    text
    start_logit: span start token logit
    end_logit: span end token logit
    char_start: start position of the character in original doc
    char_end: 
    token_start:  start position in doc tokens
    token_end
    qa_id: to locate the example
    feature_index
    prob
```
Run
```Bash
python -m cont_gen.qa_pred --max_answer_length 256 --model_outputs runs/qa/roberta-base_lr1e-4_bs16/checkpoint-12000
```

**Output**: a dict from qid (title_clause) / example to prediction in a dict
```
    all_preds
    score_null
    pred_text
```

### Get metrics
Run
```Bash
python -m cont_gen.evaluate.eval_qa_result data/cuad_clean/CUADv1_chunks_merge.jsonl \
data/clause/ori_clause_names.json \
runs/qa/roberta-base_lr1e-5_bs16/checkpoint-13000/predictions_ml256.pkl
```

## Note
answer spans can overlap.

408 for train, 102 for test
```
Features:
    train:
        examples: 16728
            has_clause: 5458
            null: 11270
        examples_expand: 22450
        
        features: 41952
    
    test: 
        examples: 4182 (102 * 41)
            has_clause: 1244
            null: 2938     
        
        features: 156623
```

## GenQA Pipeline
### Build Features
First, convert dataset into examples, each example is a paragraph of one contract, with available qas.
```Bash
source .env

python -m cont_gen.data_process.build_genqa_feature get_examples \
--data_path $CUAD_TRAIN \
--save_path ./data/genqa/train_examples.pkl
```

Tokenize paragraphs
```Bash
python -m cont_gen.data_process.build_genqa_feature tokenize \
--data_path ./data/genqa/train_examples.pkl \
--save_path ./data/genqa/train_token_ids_llama2.pkl \
--tokenizer $LLAMA2_7B \
--n_cpu 10
```