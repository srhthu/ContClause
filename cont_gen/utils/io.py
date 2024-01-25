import json

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