# Data Formats

## Original CUAD Data

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

How to download the original dataset
```Bash
pip install gdown

gdown 1of37X0hAhECQ3BN_004D8gm6V88tgZaB -O contracts.tar.gz

if [ ! -d cuad_contracts ]; then
mkdir cuad_contracts
fi

tar -xzf contracts.tar.gz -C cuad_contracts
```

## Processed Data

### data/cuad_clean/CUADv1.jsonl

```json
{
    "title": "the UID of the document",
    "doc_text": "the contract document", # 123
    "qas": [ # full list of question-answer information
        {
            "qa_id": "the UID of the document-question pair question",
            "is_impossible":  "(`bool`) True if has answers",
            "answers": [
                # a list of answers
                { "text": "answer text",
                  "start_pos/end_pos": "position of the start/end character"}
            ]
        }
    ],
    "new2old_map": "(List[int])  mapping from new to old doc character index"
}
```

Note: the **qas** contains all clause types

### data/cuad_clean/CUADv1_paras.jsonl
```json
{
    "title": "the UID of the document",
    "paras": [
        {
            "text": "paragraph text",
            "offset": "start index of the first character of the paragraph",
            "qas": [],
                # same structure as cleaned data, only keep valid qas.
                # remove is_impossible; add 'q_id' (clause type id)
        }
    ]
}
```