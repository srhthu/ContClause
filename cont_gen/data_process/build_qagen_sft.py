"""
Build supervised fine-tuning data for Generative QA
"""
from collections import Counter, OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer

from cont_gen.data_process.utils import rand_choice, ommit_middle
from cont_gen.utils import load_json, load_jsonl

Template = 'Provision:\n{provision}\n{quest}\nAnswer:\n{answers}'

def build_one_prompt(template, text, quest, answers):
    """The basic prompt format"""
    return template.format(
        provision = text,
        quest = quest,
        answers = '\n'.join(answers)
    )

def ommit_middle(text, tokenizer, max_len):
    enc = tokenizer(text)
    if len(enc['input_ids']) <= max_len:
        return text
    else:
        head_end = enc.token_to_chars(max_len // 2 - 1).end
        tail_start = enc.token_to_chars(len(enc['input_ids']) - max_len // 2).start
        return text[:head_end] + ' ... ' + text[tail_start:]

def build_prompt_with_ommit(template, text, quest, tokenizer, max_len, answers = None)->str:
    if answers is None:
        answers = ['None']
    else:
        answers = [ommit_middle(k, tokenizer, max_len) for k in answers]
    return build_one_prompt(template, text, quest, answers)

def build_chunk_prompt(chunk_data, quest_list, template, tokenizer, ommit_len):
    for qa in chunk_data['qas']:
        quest = quest_list[qa['q_id']]
        
        prompt = build_prompt_with_ommit(
            template, chunk_data['text'], quest, tokenizer, ommit_len, 
            answers = [k['text'] for k in qa['answers']]
        )

        yield {
            'chunk_index': chunk_data['chunk_index'],
            'q_id': qa['q_id'],
            'prompt': prompt
        }

def build_one_sample_training(chunks, quest_list, template, tokenizer, ommit_len):
    """
    Build prompts of chunks of one contract. Do sampling
    """
    pos_prompts = []
    for chunk in chunks:
        for p_data in build_chunk_prompt(chunk, quest_list, template, tokenizer, ommit_len):
            pos_prompts.append({**p_data, 'is_pos': True})
    
    # for negative samples
    chunk_id_to_qids = {
        i: set([qa['q_id'] for qa in chunk['qas']])
        for i, chunk in enumerate(chunks)
    }
    cla_id_counts = Counter([k['q_id'] for k in pos_prompts])
    neg_prompts = []
    for q_id, n_sample in cla_id_counts.items():
        # sample n_sample negative samples of q_id question/clause
        cdd_idx = [i for i, qids in chunk_id_to_qids.items() if q_id not in qids]
        cdd_idx = rand_choice(cdd_idx, n_sample)
        for ck_i in cdd_idx:
            neg_p = build_prompt_with_ommit(
                template, chunks[ck_i]['text'], quest_list[q_id], 
                tokenizer, ommit_len, 
                answers = None
            )
            neg_prompts.append({
                'chunk_index': ck_i,
                'q_id': q_id,
                'prompt': neg_p,
                'is_pos': False
            })
    all_prompts = pos_prompts + neg_prompts
    # will add the title in outer function
    return all_prompts

def build_training_prompt_data(
    all_cont_data, tokenizer, quests, 
    template, 
    max_answer_len, ratio = 1.0
):
    """Collect prompts of contracts with titles"""
    # group by title
    tit2chunks = {t: [] for t in titles}
    for ck in chunks:
        if ck['title'] in tit2chunks:
            tit2chunks[ck['title']].append(ck)
    all_prompts = []
    for title in tqdm(titles):
        cont_p = build_one_sample_training(
            tit2chunks[title], quest_list, template, tokenizer, ommit_len
        )
        assert len(cont_p) > 0, f'{title} do not have data'
        for d in cont_p:
            d['title'] = title
        all_prompts.extend(cont_p)
    return all_prompts

def build_test_prompt_data(
    titles, chunks, quest_list, template, 
    tokenizer, ommit_len
):
     # group by title
    tit2chunks = {t: [] for t in titles}
    for ck in chunks:
        if ck['title'] in tit2chunks:
            tit2chunks[ck['title']].append(ck)
    
    all_prompts = []
    for title in tqdm(titles):
        assert len(tit2chunks[title]) > 0, f'{title} do not have data'
        for chunk in tit2chunks[title]:
            # for each chunk, ask all questions
            for q_id, quest in enumerate(quest_list):
                pmt = build_prompt_with_ommit(
                    template, chunk['text'], quest, tokenizer, ommit_len, 
                    answers = []
                )
                all_prompts.append({
                    'title': title,
                    'chunk_index': chunk['chunk_index'],
                    'q_id': q_id,
                    'prompt': pmt
                })
    
    return all_prompts


if __name__ == '__main__':
    from pathlib import Path
    import json
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help = 'file of paragraph data')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer_name')
    parser.add_argument('ommit_len', type = int)
    parser.add_argument('--quest', help = 'file of all questions')
    parser.add_argument('--split_titles', 
                        help = 'path of a json file containing titles of the split')
    parser.add_argument('--test', action = 'store_true', 
                        help = 'True to build test data without sampling')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code = True)

    cont_data = load_jsonl(args.input_path)
    quests = load_json(args.quests)
    titles = set(load_json(args.split_titles))

    # Filter train or test data
    cont_data = list(filter(lambda k: k['title'] in titles, cont_data))

    if not args.test:
        prompts = build_training_prompt_data(
            titles, all_chunk_data, quests, Template, tokenizer, args.ommit_len
        )
    elif args.test_split:
        test_ori_data = json.load(open(args.test_split))
        titles = [k['title'] for k in test_ori_data['data']]
        prompts = build_test_prompt_data(
            titles, all_chunk_data, quests, Template, tokenizer, args.ommit_len
        )
    else:
        exit()

    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents = True, exist_ok = True)

    with open(out_p, 'w') as f:
        for d in prompts:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
