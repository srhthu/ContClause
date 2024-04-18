"""
Build supervised fine-tuning data for Generative QA.

Latested version.

Similar with build_genqa_with_context.py
"""
import random
from collections import Counter, OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Optional, Tuple

from cont_gen.data_process.utils import rand_choice, ommit_middle
from cont_gen.utils import load_json, load_jsonl, save_jsonl

DEFAULT_TEMPLATE = 'Provision:\n{provision}\n{quest}\nAnswer:\n'

def build_answer(answers: Optional[List[str]], tokenizer, max_len):
    if answers is None:
        return "None"
    return '\n'.join(ommit_middle(answer, tokenizer, max_len) for answer in answers)

def build_inputs_with_slots(
    tokenizer, clause_text: str, quest, 
    answers:Optional[List[str]], max_answer_len: int,
    template = None
):
    """Given fields, build the prompt handling truncation and answer abbrevation"""
    source_template = template if template else DEFAULT_TEMPLATE

    source = source_template.format(provision = clause_text, quest = quest)
    target = build_answer(answers, tokenizer, max_answer_len)
    return source, target

def build_prompt_data_onlyqa(
    title2paras, title: str, para_idx: int, q_id: int, quest: str,
    tokenizer, max_answer_len: int, template = None
)->Tuple[str, str]:
    """
    Build positive or negative data. 
    The answer is inferred based on whether q_id is in qas.

    Return two strings of source and target.
    """
    paras = title2paras[title]
    para_data = paras[para_idx]
    # look for answers
    answers = None
    for qa in para_data['qas']:
        if qa['q_id'] == q_id:
            answers = [k['text'] for k in qa['answers']]

    source, target = build_inputs_with_slots(
        tokenizer, para_data['text'], quest, 
        answers = answers,
        max_answer_len = max_answer_len,
        template = template
    )
    return source, target

def build_prompt_data_onlyqa_w_asw(
    title2paras, title: str, para_idx: int, q_id: int, quest: str,
    tokenizer, max_answer_len: int, template = None, answers = None
)->Tuple[str, str]:
    """
    The difference is that the answers are explicitly provided.
    """
    paras = title2paras[title]
    para_data = paras[para_idx]
    source, target = build_inputs_with_slots(
        tokenizer, para_data['text'], quest, 
        answers = answers,
        max_answer_len = max_answer_len,
        template = template
    )
    return source, target

def build_training_prompt_data(
    all_cont_data, tokenizer, quests, 
    max_answer_len, ratio = 1.0,
    template: Optional[str] = None
):
    """
    Prepare training prompts with negative sampling.

    Return samples with keys:
        title, para_idx, q_id, source, target
    """
    # 1. Get negative sample meta data
    # 1.1 Get number of positive and negative samples
    count_cla = Counter()
    for cont_data in all_cont_data:
        for para in cont_data['paras']:
            count_cla.update([qa['q_id'] for qa in para['qas']])
    print(f'Clause sample count: {count_cla}')
    num_neg_q = {q_id: int(num_pos * ratio) for q_id, num_pos in count_cla.items()}
    # 1.2 Negative sampling
    # Traverse
    all_neg_samples = []
    for q_id, num_neg in num_neg_q.items():
        cla_neg = [] # (title, para_idx, q_id)
        while len(cla_neg) < num_neg:
            cont_idx = random.choice(range(len(all_cont_data)))
            cont_data = all_cont_data[cont_idx]
            para_idx = random.choice(range(len(cont_data['paras'])))
            
            # make sure the paragraph does not contain the clause
            if q_id in [qa['q_id'] for qa in cont_data['paras'][para_idx]['qas']]:
                continue
            
            # do not sample negative samples twice
            # neg_sample = (cont_data['title'], para_idx, q_id)

            # [FixBug]
            neg_sample = (cont_data['title'], para_idx, q_id, None)

            if neg_sample in cla_neg:
                continue
            cla_neg.append(neg_sample)
        all_neg_samples.extend(cla_neg)
    
    # 2. Get positive sample meta data
    title2paras = {d['title']: d['paras'] for d in all_cont_data}
    all_pos_samples = []
    for title, paras in title2paras.items():
        for pi, para in enumerate(paras):
            for qa in para['qas']:
                # all_pos_samples.append((title, pi, qa['q_id']))
                # [FixBug]
                answers = [k['text'] for k in qa['answers']]
                all_pos_samples.append((title, pi, qa['q_id'], answers))
    
    total_samples = all_pos_samples + all_neg_samples
    print(
        f'pos samples: {len(all_pos_samples)}  '
        f'neg samples: {len(all_neg_samples)}'
    )

    # 3. Get source and target
    prompt_data = []

    for title, para_idx, q_id, answers in tqdm(total_samples, ncols = 80):
    # for title, para_idx, q_id in tqdm(total_samples, ncols = 80):
        # source, target = build_prompt_data_onlyqa(
        #     title2paras, title, para_idx, q_id, quests[q_id], 
        #     tokenizer,
        #     max_answer_len = max_answer_len,
        #     template=template
        # )

        # [FixBug]
        source, target = build_prompt_data_onlyqa_w_asw(
            title2paras, title, para_idx, q_id, quests[q_id], 
            tokenizer,
            max_answer_len = max_answer_len,
            template=template,
            answers = answers
        )

        prompt_data.append({
            'title': title,
            'para_idx': para_idx,
            'q_id': q_id,
            'source': source,
            'target': target
        })
    return prompt_data

def build_test_prompt_data(
    all_cont_data, tokenizer, quests, 
    max_answer_len,
    template: Optional[str] = None
):
    """
    Prepare test prompts with all questions
    """
    title2paras = {d['title']: d['paras'] for d in all_cont_data}
    all_samples = []
    for title, paras in title2paras.items():
        for pi, para in enumerate(paras):
            for q_id in range(len(quests)):
                all_samples.append((title, pi, q_id))
    
    
    # 3. Get source and target
    prompt_data = []
    for title, para_idx, q_id in tqdm(all_samples, ncols = 80):
        source, target = build_prompt_data_onlyqa(
            title2paras, title, para_idx, q_id, quests[q_id], 
            tokenizer,
            max_answer_len = max_answer_len,
            template=template
        )
        prompt_data.append({'title': title,
                            'para_idx': para_idx,
                            'q_id': q_id,
                            'source': source,
                            'target': target})
    return prompt_data


if __name__ == '__main__':
    from pathlib import Path
    import json
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help = 'file of paragraph data')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer_name')
    parser.add_argument('--ommit_len', type = int)
    parser.add_argument('--quests', help = 'file of all questions')
    parser.add_argument('--split_titles', 
                        help = 'path of a json file containing titles of the split')
    parser.add_argument('--test', action = 'store_true', 
                        help = 'True to build test data without sampling')
    parser.add_argument('--neg_ratio', type = float, default = 1.0, 
                        help = 'ratio of neg samples to pos samples')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code = True)

    cont_data = load_jsonl(args.input_path)
    quests = load_json(args.quests)
    titles = set(load_json(args.split_titles))

    # Filter train or test data
    cont_data = list(filter(lambda k: k['title'] in titles, cont_data))

    if not args.test:
        # use the default template
        prompts = build_training_prompt_data(
            cont_data, tokenizer, quests, 
            max_answer_len = args.ommit_len, ratio = args.neg_ratio,
        )
    else:
        prompts = build_test_prompt_data(
            cont_data, tokenizer, quests, 
            max_answer_len=args.ommit_len
        )

    save_jsonl(prompts, args.output_path)
