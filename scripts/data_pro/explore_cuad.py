"""Load and explore the cuad dataset"""
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
# %%
# some variables of local paths
ENVS = dotenv_values('../.env')
cuad_dir = Path(ENVS['CUAD_PATH'])
# %%
llama_tk = AutoTokenizer.from_pretrained(ENVS['LLAMA2_7B'])
# %%
class CUAD_Reader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
# %%
def get_dir_files(path):
    return list(Path(path).glob("*"))

# %%
contracts = [{"name": fn.stem, "content": open(fn).read()} for fn in get_dir_files(cuad_dir / 'full_contract_txt')]

master_clauses = pd.read_csv(cuad_dir / 'master_clauses.csv')
# %%
qa_clauses = json.load(open(cuad_dir / 'CUAD_v1.json'))
# %%
# count the input length
cont_lens = [len(llama_tk.tokenize(k['content'])) for k in contracts]
# %%
_ = plt.hist(cont_lens, bins = 50)
# %%
_ = plt.boxplot(cont_lens)
# %%
# convert filename to match the `title` in squad json file
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
if 'name' not in master_clauses.columns:
    master_clauses.insert(0, column = 'name', value = None)
master_clauses['name'] = master_clauses['Filename'].apply(convert_filename)

# %%
contracts_map = {k['name']: k['content'] for k in contracts}
all_clauses = master_clauses.columns[2:][::2]

# %%
# assert each contract title in josn file can be found
for _, line in master_clauses.iterrows():
    # assert remove_suffix(line['Filename']) in contracts_map, line['Filename']
    assert line['name'] in contracts_map, line['name']
# %%
def display_column(col):
    """Show some examples of col and col answer"""
    df = master_clauses[[col, col + '-Answer']]
    part = df.sample(n=5)
    for _, row in part.iterrows():
        print(f'{col}: {row.iloc[0]}\nAnswer: {row.iloc[1]}')
# %%
display_column(all_clauses[1])

# %%
qa_data = qa_clauses['data']
"""
title: contract id
paragraphs: a list of length 0
    context
    qas: a list of question answer dict:
        answers: a list of answer text and position dict:
            text: text span in original contract
            answer_start: int
        id: <title>__<clause type>
        question: a template contain the clause type
        is_impossible: True if there is no answer
"""
# %%
assert contracts_map[qa_data[0]['title']] == qa_data[0]['paragraphs'][0]['context']
# %%
sample_id = qa_data[0]['title']
qas = qa_data[0]['paragraphs'][0]['qas']

imp_qas = [k for k in qas if k['is_impossible']]
# %%
master_clauses[master_clauses['Filename'] == (sample_id + '.PDF')][[
    'Notice Period To Terminate Renewal',
    'Notice Period To Terminate Renewal' + '- Answer']]
# %%
list(master_clauses.columns).index('Notice Period To Terminate Renewal')
master_clauses.columns[12:15]
# %%
k = 454
all_filenames = list(master_clauses.Filename)
sample_id = qa_data[k]['title']
print([(i,k) for i,k in enumerate(all_filenames) if sample_id[-20:].lower() in k.lower()])
# %%
print([k for k in all_filenames if not k.lower().endswith('pdf')])
# %%
# to check qas in squad data match the main csv file
misses = []
for i, s_d in enumerate(qa_data):
    sample_id = s_d['title']
    qas = s_d['paragraphs'][0]['qas']
    assert contracts_map[sample_id] == s_d['paragraphs'][0]['context']
    df_line = master_clauses[master_clauses['name'] == sample_id.strip().removesuffix('-')]
    if not len(df_line) > 0:
        misses.append((i, sample_id))
    continue
    for c_qa in qas:
        c_type = c_qa['id'].split('__')[1]
        assert c_type in master_clauses.columns, c_type
        span1 = eval(df_line[c_type].iloc[0])
        span2 = [k['text'] for k in c_qa['answers']]
        assert len(span1) == len(span2) and all([k in span2 for k in span1]), (i, c_type, span1, span2)
# %%
for i, s_d in enumerate(qa_data):
    sample_id = s_d['title']
    if sample_id.startswith('MSCIINC_02_28_2008'):
        print(sample_id)
# %%
text = contracts_map[qa_data[0]['title']]
# %%
text.index('Term of the  Agreement  and for a period of  eighteen')
# %%
text[44904-100:44904+100]
text_json = qa_data[0]['paragraphs'][0]['context']
# %%
print(text_json == text)
# %%
# no overlap of clauses checking
for i, s_d in enumerate(qa_data):
    sample_id = s_d['title']
    text =s_d['paragraphs'][0]['context'] 
    qas = s_d['paragraphs'][0]['qas']
    # token clause types
    seq_type = [[] for _ in text]
    for c_qa in qas:
        c_type = c_qa['id'].split('__')[1]
        for asw in c_qa['answers']:
            start = asw['answer_start']
            end = start + len(asw['text'])
            for k in range(start, end):
                seq_type[k].append(c_type)
    print(len([k for k in seq_type if len(k) > 1]))
# %%
sample = qa_data[15]
para = sample['paragraphs'][0]
doc = para['context']
# %%
idx = [i for i, ans in enumerate(para['qas']) if 'Warranty' in ans['question']][0]
idx = [i for i, ans in enumerate(para['qas']) if 'Ip Ownership Assignment' in ans['question']][0]
# %%
clauses = [re.search(r'\"(.*?)\"', q['question']).group(1) for q in qa_data[0]['paragraphs'][0]['qas']]
# %%
c_id = [i for i,c in enumerate(clauses) if 'warranty' in c.lower()][0]
c_id = clauses.index('Ip Ownership Assignment')
# %%
part = [d for d in qa_data if not d['paragraphs'][0]['qas'][c_id]['is_impossible']]
print(len(part))
# %%
doc = part[0]['paragraphs'][0]['context']
qa = part[0]['paragraphs'][0]['qas'][c_id]
print(qa)
# %%
def clean_space(s):
    return re.sub(r'([ ])+', ' ', s)
# %%
answers = sorted(qa['answers'], key = lambda k: k['answer_start'])
for a in answers:
    a['text'] = clean_space(a['text'])
# %%
print(clean_space(doc[answers[0]['answer_start']-500: answers[0]['answer_start']+1000]))
# %%
