"""Refine the questions for each clause"""
# %%
import json
import re
from transformers import AutoTokenizer
# %%
with open('../../data/cuad_clean/CUADv1.jsonl') as f:
    clean_data = [json.loads(k) for k in f]

with open('../../data/clause/clause_info.json') as f:
    cla_info = json.load(f)
# %%
quests = [k['question'] for k in clean_data[0]['qas']]
print(quests[0])
# %%
# output different category names
cat_extracted = [re.search(r'related to "(.+)"', q).group(1) for q in quests]
for cinfo, n2 in zip(cla_info, cat_extracted):
    if cinfo['category'] != n2:
        print((cinfo['category'],n2))
# There are a few mismatched clause types between readme and quest in data...
# %%
# output different category details
detail_ext = [re.search(r'Details: (.+)', q).group(1) for q in quests]
detail_ext = [re.sub('\xa0','', k) for k in detail_ext]
for cinfo, n2 in zip(cla_info, detail_ext):
    if cinfo['desc'] != n2:
        print((cinfo['desc'],n2))
# There are a few mismatched clause information...
# %%
# Prompt Template
tplt_quest = 'Question: Is there any part related to (clause) "{name}"?'
tplt_quest_desc = (
    'Question: Is there any part related to (clause) "{name}"?\n'
    'Description: {desc}'
)
tplt_desc_quest = (
    'Knowledge: {name} is about "{desc}"\n'
    'Question: Is there any part related to (clause) "{name}"?'
)
# %%
prompt_quest = [tplt_quest.format(name = name, desc = desc) 
                for name, desc in zip(cat_extracted, detail_ext)]
prompt_quest_desc = [tplt_quest_desc.format(name = name, desc = desc) 
                for name, desc in zip(cat_extracted, detail_ext)]
# %%
with open('../data/clause/prompt_quest.json', 'w') as f:
    json.dump(prompt_quest, f, indent=4, ensure_ascii=False)
# %%
with open('../data/clause/prompt_quest_desc.json', 'w') as f:
    json.dump(prompt_quest_desc, f, indent=4, ensure_ascii=False)
# %%
prompt_desc_quest = [tplt_desc_quest.format(name = name, desc = desc) 
                for name, desc in zip(cat_extracted, detail_ext)]
print(prompt_desc_quest[0])
with open('../../data/clause/prompt_desc_quest.json', 'w') as f:
    json.dump(prompt_desc_quest, f, indent=4, ensure_ascii=False)
# %%
