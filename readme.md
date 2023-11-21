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

## Data Process Pipeline
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
```Bash
# for debug
python -m cont_gen.infer_qa --save_path runs/debug/model_outputs.pkl --ckpt roberta-base --features data/features/qa_roberta_test.pkl

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
Given all prelim predictions, filter top ranking predictions, and get the answer span text and position
```
    text
    start_logit
    end_logit
    char_start
    char_end
    token_start # start position in doc tokens
    token_end
    qa_id: to locate the example
    prob
```


## Note
answer spans can overlap.

408 for train, 102 for test

Features:
    train: 
    test: 156623