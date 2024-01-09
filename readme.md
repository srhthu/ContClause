## CUAD Data
Data Format of the `CUAD_v1.json`:

```
"version":
"data": a list of contract json data
    "title": contract id. It does not totally match the file name
    "paragraphs": only one paragraph for each contract.
        "context": the contract content
        "qas": a list of question answer dict:
            "answers": a list of answer text and position dict:
                text: text span in original contract
                answer_start: int
            id: <title>__<clause type>
            question: a template contain the clause type
            is_impossible: True if there is no answer
```

### Download extra contract data
```Bash
pip install gdown

gdown 1of37X0hAhECQ3BN_004D8gm6V88tgZaB -O contracts.tar.gz

if [ ! -d cuad_contracts ]; then
mkdir cuad_contracts
fi

tar -xzf contracts.tar.gz -C cuad_contracts
```

# Pipeline for Generative Methods
## Overview
First, we pre-process the original dataset to [Link](#text-pre-process)
- remove extra spaces and newlines
- sort answer spans and merge overlapping spans.
- convert to a concise structure

Then, we split documents into paragraphs and relocate the answers of each paragraph. [Link](#split-paragraphs)

Next, we **merge short paragraphs** with the next one. [Link](#merge-short-paragraphs)
- short paragraphs can be chapter titles
- If len(cur_p) < thresh 1 ; then merge it with the next paragraph
- Repeat the process

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

## Data  Process
### Text Pre-process
**Code**: `data_process/pre_process.py`

**Run**: 
```
python -m cont_gen.data_process.pre_process data/cuad_split/CUADv1.json data/cuad_clean/CUADv1.jsonl
```

**Input**: original SQUAD data format

**Output**:
output to `data/cuad_clean/CUADv1.jsonl` in such format
```json
title: the UID of the document
doc_text: the contract document
qas: a list of question-answer information
    qa_id: the UID of the document-question pair
    question
    is_impossible (`bool`): True if has answers
    answers: a list of answers
        text: answer text
        start_pos: position of the start character
        end_pos: position of the end character
new2old_map: (List[int])
```

### Split paragraphs
**Code**: `data_process/split_para.py`

**Aim**:
- Split by paragraph
- Find QAs in this paragraph and relocate the answer position

**Run**
```
python -m cont_gen.data_process.split_para data/cuad_clean/CUADv1.jsonl data/cuad_clean/CUADv1_paras.jsonl
```
**Input**
`data/cuad_clean/CUADv1.jsonl`

**Output**

`data/cuad_clean/CUADv1_paras.jsonl`: A list of contract samples. Each sample contains several paragraphs.
```
title: the UID of the document
paras: a list of paragraphs
    text: paragraph text
    offset: start index of the first character
    qas: a list of **available** QA pairs
        qa_id: UID of form {title}_{quest}_{span_id}
        q_id: index of question.
        answers: a list of answers
            text: answer text
            start_pos: position of the start character
            end_pos: position of the end character
```

### Merge short paragraphs
**Code**: `data_process/merge_short.py`

**Run**
```Bash
python -m cont_gen.data_process.merge_short data/cuad_clean/CUADv1_paras.jsonl data/cuad_clean/CUADv1_paras_merge.jsonl
```

**Input**: paragraph data, `data/cuad_clean/CUADv1_paras.jsonl`

**Output**: same format with merged paragraph data. `data/cuad_clean/CUADv1_paras_merge.jsonl`

Note: the average number of paragraphs change from **145.97** to **61.37**; the average length of paragraphs increase from **344.89** to **821.63**

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

## Training
training code: `cont_gen/train_genqa.py`

Run the script: 
- `sh/genqa_phi15_ds.sh`: use deepspeed zero stage 1 to train the model

## Evaluating

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

Output:
```
    all_preds
    score_null
    pred_text
```

### Get metrics


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