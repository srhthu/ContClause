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

### OOD Meta data

Code: `cont_gen/data_process/ood/build_sft_meta_data.py`

Output dir:`data/ood_split/seed{seed}_tr{num_tr_labels}`
- `train_meta.csv`: 'title', 'para_idx', 'q_id', 'answers', 'type'
  - type: 0: positive, 1: negative (quest2clause), 2: negative (pos_clause2quest)
- `test_meta_ood.csv`: 'title', 'para_idx', 'q_id', 'answers', 'small'
  - small: value of [0 ,1, 2, 3], small set with value >0 
    - 1: positive, 2: negative (para->all quest), 3: negative(quest -> sample para)
- `train_labels.csv`: clause_id, clause_type
- `test_labels.csv`: clause_id, clause_type

### SFT Data

Given the meta data, we truncate based on LLama3's tokenizer to a max length of 512

Output dir: `data/ood_split/<split_dir>/<prompt_name>/`
- `train_data.json`
- `test_data_id.json`
- `test_data_ood.json`