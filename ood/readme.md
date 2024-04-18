## Pre-process Data Formats

### Clause information
Path: `data/clause/all_info.csv`
Columns:
- **clause_type**
- **clause_quest**
- **readme_clause_type**
- **readme_group**
- **readme_aws_format**

### Contract Paragraph data

Path: `data/cuad_clean/CUADv1_paras_merge_new.jsonl`
```Json
"title": "the UID of the document",
    "paras": [
        {
            "text": "paragraph text",
            "offset": "start index of the first character of the paragraph",
            "qas":[
                "q_id": "clause global id",
                "answers": [
                    {
                        "text": "answer text",
                        "start_pos": "(int)",
                        "end_pos": "(int)"}
                ]
            ],
        }
    ]
```

## Process OOD Data

We aim to segment the training and test set based on disjoint labels.

First, we generate instruction meta data, including the original input and output without prompt. We apply negative sampling strategy here.

Then, we embed prompts and do some modification of the answer.

### 