# Overview of Repo
The repo contains the following folders:
- `config`: config file for deepspeed or prompts
- `data`: dataset and processed data
- `cont_gen`: the library for the whole project
- `ood`: scripts for the OOD settihng
- `pretrain`: old code to pre-train the model
- `runs`: experiment results, e.g., checkpoints
- `scripts`: script for data process, visualization, data exploration, debug ...
  - `analyze`: scripts to analyze model outputs
- `sh`: bash scripts
- `cache`: caches for dataset features

### Data
We introduce how the data under `data` folder is generated.

**clause**

This folder contains meta clause information and prompts.

**cuad_clean**

The processed CUAD dataset.

`CUADv1.json`: Basic clean [clean](#clean-and-format)

`CUADv1_paras.jsonl`: Split based on paragraphs. [split](#split-documents-into-paragraphs)

`CUADv1_paras_merge_new.jsonl`: Merge short paragraphs. [merge](#merge-short-paragraphs)
- ~~`CUADv1_paras_merge.jsonl`~~ is the old wrong version.

`merge_split` folder: Split long paragraphs for each tokenizer. [split_long](#split-long-paragraphs)

**ood_split**

The training and test data for the OOD setting. Every tokenizer has its own data under a specific split. The data contains the meta data and prompt data that is used for model training and inference.

Code: `scripts/jupyter/data_pro/build_sft.ipynb`

**cuad_sft**

The training and test data for the task. Similar to **ood_split** but training and test cover all labels.

### Analyze

`ood_preds.ipynb`: analyze the ood and id prediction errors.

# Pre-Process
Before method-specific processing, the original dataset need be cleaned. We will also build some data that will be further used by various methods.

## Clean and Format
First, we pre-process the original dataset:
- Reduce **consecutive spaces and newlines** to one
- **Sort** answer spans and merge overlapping spans.
- Convert to a **concise structure**

**Code**: `data_process/pre_process/clean.py`

**Run**: 
```
python -m cont_gen.data_process.pre_process.clean data/cuad_split/CUADv1.json data/cuad_clean/CUADv1.jsonl
```

**Output**: `data/cuad_clean/CUADv1.jsonl`. [format](doc/data_formats.md#datacuad_cleancuadv1jsonl)

## Split Documents into Paragraphs
Split documents into paragraphs and **relocate** the answers of each paragraph and only keep valid questions.

**Code**: `data_process/pre_process/split_doc_to_para.py`

**Run**
```Bash
python -m cont_gen.data_process.split_doc_to_para \
data/cuad_clean/CUADv1.jsonl \
data/cuad_clean/CUADv1_paras.jsonl
```

**Output**: `data/cuad_clean/CUADv1_paras.jsonl`. [format](data_formats.md#datacuad_cleancuadv1_parasjsonl)

## Merge Short Paragraphs
Mmerge short paragraphs with its successors **heuristically**.
  
**Code**: `data_process/pre_process/merge_short_new.py`  

**Run**
```Bash
python -m cont_gen.data_process.merge_short_new \
data/cuad_clean/CUADv1_paras.jsonl \
data/cuad_clean/CUADv1_paras_merge.jsonl \
--threshold 300
```

**Output**: same format of paragraph data. `data/cuad_clean/CUADv1_paras_merge_new.jsonl`

Note: the average number of paragraphs per doc change from **145.97** to **61.37**; the average length of paragraphs increase from **344.89** to **821.63**.

Legency: old version script is `data_process/pre_process/merge_short.py` and old data is `data/cuad_clean/CUADv1_paras_merge.jsonl`.

## Split Paragraphs to Max Token Length (Legency)
<!-- Split paragraphs to make each one not exceed max number of tokens for a **tokenizer**.

**Code**: `data_process/split_para_max_token.py`

Arguments: input_path, output_path, tokenizer, max_para_len

**Output** to `data/cuad_clean/{tokenizer_name}_{max_para_len}/CUAD_paras.jsonl`

**Statistics**: there are total (train + test) 33743 paragraphs, only 6531 (19.36\%) paragraphs contain pre-defined clauses.  -->

## Split Long Paragraphs
**Aim**: GPU memory limitation requires limited input sequence length.

**Script**: 
- `scripts/jupyter/data_pro/split_long_para.ipynb` for excute
- `cont_gen/data_process/pre_process/split_long_para.py`

**Output**: `data/cuad_clean/merge_split/paras_{tk_name}_512.jsonl`





