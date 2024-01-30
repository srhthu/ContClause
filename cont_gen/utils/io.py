import json
import pickle

def load_jsonl(path):
    with open(path) as f:
        data = [json.loads(k) for k in f]
    return data

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_jsonl(data, path):
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
    with open(path, 'wb') as f:
        pickle.dump(data, f)