"""
Build the source and target for Generative QA model with context.

The starting point is the paragraph data, e.g., data/cuad_clean/CUADv1_paras_merge.jsonl
"""
from collections import Counter, OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Optional, Any
import random

from cont_gen.data_process.utils import tokenize_wo_eos, ommit_middle, overlap_of_two_span
from cont_gen.data_process.split_para_to_chunks import chunk_text


def build_answer(answers: Optional[List[str]], tokenizer, max_len):
    if answers is None:
        return "None"
    return '\n'.join(ommit_middle(answer, tokenizer, max_len) for answer in answers)

def build_inputs_with_both(
    tokenizer,
    clause_text, 
    quest,
    answers,
    memory_head: List[str],
    memory_clauses: List[List[str]],
    total_memory_len: int,
    max_answer_len: int
):
    source_template = 'Previous clauses:\n{memory}\nProvision:\n{provision}\n{quest}\nAnswer:\n'
    # first, we prepare the memory of each paragraph
    mem_tpl = 'Heading: {head}. This paragraph specifies the clauses: {cla_names}.\n'
    memory_list = [mem_tpl.format(
        head = head, 
        cla_names = ', '.join(cla_names)
        ) for head, cla_names in zip(memory_head, memory_clauses)]
    # find the max number of memorys to fit into the window size
    mem_lens = [len(tokenize_wo_eos(text).inpu_ids) for text in memory_list]
    mem_window = 0
    mem_acc_len = 0
    while mem_acc_len <= total_memory_len and mem_window < len(memory_list):
        mem_window += 1
        mem_acc_len += mem_lens[-mem_window]
    memory_text = ''.join(memory_list[-mem_window:])
    # deal with no memory
    if memory_text == '':
        memory_text = 'None'
    source = source_template.format(
        memory = memory_text, provision = clause_text, quest = quest
    )
    target = build_answer(answers, tokenizer, max_answer_len)
    return source, target

def get_head(text: str, num_tokens = 5):
    """
    Given a paragraph, get the headings with heuristic rules
    """
    heads = []
    lines = text.split('\n')
    for line in lines:
        if len(line) == 0:
            continue
        first_sent = ' '.join(line.split(' ')[:num_tokens])
        heads.append(first_sent)
    return '. '.join(heads)
    

def get_memory_data(data, clause_names: List[str], num_head_tokens = 5):
    """
    Given a contract sample of all paragraphs, 
    get mem_head and mem_clauses for each paragraph as context metadata.
    """
    # extract context data for each paragraph
    for para_data in data['paras']:
        para_data['mem_head'] = get_head(para_data['text'], num_tokens = num_head_tokens)
        para_data['mem_clauses'] = [clause_names[qa['q_id']] for qa in para_data['qas']]

def filter_and_relocate_answer(answer, start, end, text):
    """
    Return answer in [start, end] or None if no overlapping.

    Args:
        answer: contain the position in paragraph text
        start, end: position of current span
        text: paragraph text
    """
    ol_pos = overlap_of_two_span((start, end), (answer['start_pos'], answer['end_pos']))
    if ol_pos is None:
        return None
    return {
        'text': text[ol_pos[0]: ol_pos[1] + 1],
        'start_pos': ol_pos[0] - start,
        'end_pos': ol_pos[1] - start
    }

def split_para_to_chunk_para(cont_data, tokenizer, max_para_len: int):
    """
    Split paragraphs into chunk_paragraph to fit in the context size
    """
    new_paras = []
    for para in cont_data['paras']:
        chunk_pos = chunk_text(para['text'], tokenizer, max_len = max_para_len)
        for start, end in chunk_pos:
            # prepare the new paragraph
            new_p = {
                'text': para['text'][start: end + 1],
                'offset': para['offset'] + start,
                'qas': []
            }
            for qa in para['qas']:
                # find answers in current span. None if not in
                new_answers = [
                    filter_and_relocate_answer(answer, start, end, new_p['text']) for answer in qa['answers']]
                new_answers = list(filter(lambda k: k is not None, new_answers))
                if len(new_answers) == 0:
                    continue
                # build new qa data
                new_qa = {
                    'qa_id': qa['qa_id'],
                    'q_id': qa['q_id'],
                    'answers': new_answers
                }
                new_p['qas'].append(new_qa)
            new_paras.append(new_p)
    return {
        'title': cont_data['title'],
        'paras': new_paras
    }

    
def build_train_data(
    all_cont_data, tokenizer, max_para_len, quests,
    total_mem_len, max_answer_len,
    ratio = 1.0,
):
    """
    Training data with sampling of negative samples.

    Sample strategy: for each clause type, determin num_neg as num_pos * ratio,
    then sample contract and paragraphs to meet this number
    """
    print(f'Truncate paragraphs to no more than length of {max_para_len}')
    all_cont_data = [
        split_para_to_chunk_para(k, tokenizer, max_para_len) for k in all_cont_data]
    
    # Sample negative samples
    ## First, count num_pos for each clause type
    count_cla = Counter()
    for cont_data in all_cont_data:
        for para in cont_data['paras']:
            count_cla.update([qa['q_id'] for qa in para['qas']])
    
    ## get negtive samples in the form (title, para_idx, q_id)
    all_neg_samples = []
    for q_id, num_pos in count_cla.items():
        cla_neg = []
        num_neg = int(num_pos * ratio)
        while len(cla_neg) < num_neg:
            cont_idx = random.choice(range(len(all_cont_data)))
            cont_data = all_cont_data[cont_idx]
            para_idx = random.choice(range(len(cont_data['paras'])))
            # make sure the paragraph does not contain the clause
            if q_id in [qa['q_id'] for qa in cont_data['paras'][para_idx]['qas']]:
                continue
            neg_sample = (cont_data['title'], para_idx, q_id)
            if neg_sample in cla_neg:
                continue
            cla_neg.append(neg_sample)
        all_neg_samples.append(cla_neg)

    # Prepare memory data for each paragraph
    for cont_data in all_cont_data:
        get_memory_data(cont_data)

    # get pos train samples in the same form
    all_pos_samples = []
    for title, paras in title2paras.items():
        for pi, para in enumerate(paras['paras']):
            for qa in para['qas']:
                all_pos_samples.append((title, pi, qa['q_id']))

    ## Prepare prompts
    title2paras = {d['title']: d['paras'] for d in all_cont_data}
    
    for title, para_idx, q_id in all_pos_samples + all_neg_samples:
        build_prompt_data(
            title2paras, title, para_idx, q_id, quests[q_id], 
            tokenizer,
            max_mem_num = 10, total_mem_len = total_mem_len,
            max_answer_len = max_answer_len                  
        )

def build_oracle_test_data(
    all_cont_data, tokenizer, max_para_len, quests,
    total_mem_len, max_answer_len,
    ratio = 1.0,
):
    """
    The oracle test 
    """
    
def build_prompt_data(
    title2paras, title, para_idx, q_id, quest,
    tokenizer,
    max_mem_num, total_mem_len, max_answer_len
):
    """
    Build positive or negative data infered by whether q_id is in qas
    
    Return: a dict of title, para_idx, q_id, source, target
    """
    paras = title2paras[title]
    para_data = paras[para_idx]
    # look for answers
    answers = None
    for qa in para_data['qas']:
        if qa['q_id'] == q_id:
            answers = [k['text'] for k in qa['answers']]
    
    mem_first = max(para_idx - max_mem_num, 0)
    mem_heads = [k['mem_head'] for k in paras[mem_first: para_idx]]
    mem_clauses = [k['mem_clauses'] for k in paras[mem_first: para_idx]]

    source, target = build_inputs_with_both(
        tokenizer, para_data['text'], quest, answers = answers,
        memory_head = mem_heads, memory_clauses = mem_clauses,
        total_memory_len = total_mem_len,
        max_answer_len = max_answer_len
    )
    return source, target

if __name__ == '__main__':
    from pathlib import Path
    import json
    import argparse
    
    parser = argparse.ArgumentParser()
