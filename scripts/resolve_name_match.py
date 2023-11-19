"""Resolve the issue of mismatch file name in the directory, csv file and json file"""
# %%
import json
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import re
import numpy as np
from pprint import PrettyPrinter
from IPython.lib.pretty import pprint
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from dotenv.main import dotenv_values
from typing import List
from difflib import SequenceMatcher
# %%
# some variables of local paths
ENVS = dotenv_values('../.env')
cuad_dir = Path(ENVS['CUAD_PATH'])
# %%
def get_dir_files(path):
    return list(Path(path).glob("*"))
# %%
contracts = [{"name": fn.stem, "content": open(fn).read()} for fn in get_dir_files(cuad_dir / 'full_contract_txt')]
contracts_map = {k['name']: k['content'] for k in contracts}

master_clauses = pd.read_csv(cuad_dir / 'master_clauses.csv')
qa_clauses = json.load(open(cuad_dir / 'CUAD_v1.json'))

# %%
class StrMatch:
    """Given a corpus and a query, find the best match one"""
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
    
    def rank(self, x):
        """Return the corpus elements ranking on the similarity with x"""
        scores = [self.get_ratio(x, c) for c in self.corpus]
        rank_idx = np.flip(np.argsort(scores))
        return [self.corpus[k] for k in rank_idx]

    def match(self, x):
        """Return the best match item in corpus"""
        return self.rank(x)[0]
    
    def get_ratio(self, a, b):
        """Get the similarity ratio"""
        return SequenceMatcher(None, a, b).ratio()
# %%
file_name_match = StrMatch([k['name'] for k in contracts])
# %%
def get_miss(list_a,list_b):
    """Return elements in list_a that are not in list_b"""
    return [k for k in list_a if k not in list_b]

# String process functions
def remove_suffix(text):
    return '.'.join(text.split('.')[:-1])

def convert_filename(s):
    # remove suffix of .pdf .PDF .PDF'
    s = '.'.join(s.split('.')[:-1])
    # remove space in the end
    s = s.strip()
    # replace & with _
    s = re.sub(r"[&\']", '_', s)
    # remove last character if it is -
    s = s.removesuffix('-')
    return s
# %%
miss_csv = get_miss(master_clauses['Filename'].apply(remove_suffix).to_list(), contracts_map)
# %%
for name in miss_csv:
    r = file_name_match.match(name)
    print(f'{"CSV":10}: {name}\n{"filename":10}: {r}')
# There are some rules to transform the csv name to filename
# 1. "&", "'" -> "_", this is only done on csv name
# 2. remove the space at the end of the csv name and filename
# 3. remove the last character of "-" of both csv and filename
# %%
miss_csv_2 = get_miss(master_clauses['Filename'].apply(convert_filename).to_list(), contracts_map)
print(len(miss_csv_2))
# %%
for name in miss_csv_2:
    r = file_name_match.match(name)
    print(f'{"CSV":10}: {name}\n{"filename":10}: {r}')
# %%
# assert title in json match the csv name
miss_json_csv = get_miss(
    [d['title'] for d in qa_clauses['data']],
    master_clauses['Filename'].apply(remove_suffix).to_list()
)
print(len(miss_json_csv))
# %%
miss_json_fn = get_miss(
    [d['title'] for d in qa_clauses['data']],
    contracts_map
)
"""
There is only one mismatch between title in json and filename.
"""
# %%
t = file_name_match.match(miss_json_fn[0])
# %%
