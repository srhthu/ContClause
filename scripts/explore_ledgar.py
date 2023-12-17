"""Load and explore the LEDGAR dataset"""
# %%
import json
from pathlib import Path
from collections import OrderedDict, Counter
import pandas as pd
import re
import numpy as np
from pprint import PrettyPrinter
from IPython.lib.pretty import pprint
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from dotenv.main import dotenv_values
# %%
# some variables of local paths
ENVS = dotenv_values('../.env')
data_dir = Path(ENVS['LEDGAR_DIR'])
# %%
clean_data = [json.loads(k) for k in open(data_dir / 'LEDGAR_2016-2019_clean.jsonl')]
raw_data = [json.loads(k) for k in open(data_dir / 'sec_corpus_2016-2019.jsonl')]
# %%
len(clean_data)
# %%
exa1 = clean_data[0]
# %%
def load_all_doc_id(path):
    """Return a list of tuple (year, qtr, file_id)"""
    all_file_path = list(Path(path).glob("*/*/*/*.html"))
    infos = [html_path.parts[-4:-1] for html_path in all_file_path]
    acc_id = {k[-1]: k for k in infos}
    return acc_id
# %%
acc_map = load_all_doc_id(data_dir / 'data')
cik_ids = [k[:10] for k in acc_map] # first 10 digit is cik
print(len(set(cik_ids)))

cik_ct = Counter(cik_ids)
print(cik_ct.most_common(20))

rec_of_ent = [k for k in acc_map if k.startswith('000119312516')]
print(len(rec_of_ent))
# %%
# Statistics of labels
labels = [lab for k in clean_data for lab in k['label']]
label_ct = Counter(labels)
print(f'Num of labels: {len(label_ct)}')
# %%
label_rank = label_ct.most_common()
# %%
for i in range(20):
    print(label_rank[i])
# %%
clean_df = pd.DataFrame(clean_data)
# %%
part = clean_df[clean_df.apply(lambda k: 'terminations' in k['label'], axis = 1)]
# %%
len(part)
# %%
