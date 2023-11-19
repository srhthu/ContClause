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
Each feature comprise a segment (window), a question and the answer (span position).
