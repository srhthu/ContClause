import json
import pickle
from pathlib import Path
from typing import List

def make_parent_dir(path):
    """Make the parent dir of path"""
    Path(path).parent.mkdir(parents = True, exist_ok=True)

def load_jsonl(path):
    with open(path) as f:
        data = [json.loads(k) for k in f]
    return data

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_jsonl(data: List, path):
    """Save each element of data to path as json string. Create parent dir if not exist"""
    make_parent_dir(path)
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def save_jsonl_append(data, path):
    """Append json data to existing path"""
    with open(path, 'a', encoding = 'utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, path):
    make_parent_dir(path)
    with open(path, 'wb') as f:
        pickle.dump(data, f)