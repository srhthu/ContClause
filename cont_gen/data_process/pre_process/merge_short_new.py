"""
Merge short paragraphs. Fix the bug that one clause type can appear many times. 
"""

from typing import List

def merge_two_paragraphs(para_1, para_2):
    new_text = para_1['text'] + '\n' + para_2['text']
    new_offset = para_1['offset']
    p2_offset = len(para_1['text']) + 1 # 1 for \n

    p2_qas = [{
        'qa_id': qa['qa_id'],
        'q_id': qa['q_id'],
        'answers': [
            {
                'text': answer['text'],
                'start_pos': answer['start_pos'] + p2_offset,
                'end_pos': answer['end_pos'] + p2_offset
             } for answer in qa['answers']
        ]
    } for qa in para_2['qas']]

    
    qid2aws = {}
    for qa in para_1['qas'] + p2_qas:
        qid = qa['q_id']
        if qid not in qid2aws:
            qid2aws[qid] = qa['answers']
        else:
            qid2aws[qid] += qa['answers']
    
    new_qas = [{'q_id': q_id, 'answers': answers} for q_id, answers in qid2aws.items()]
    new_qas.sort(key = lambda k: k['q_id'])


    new_para = {
        'text': new_text,
        'offset': new_offset,
        'qas': new_qas
    }
    return new_para


def merge_paragraphs_and_relocate(
    para_list, 
    short_length = 300
):
    i = 0
    para_list = [k for k in para_list]
    while i < len(para_list) - 1:
        if len(para_list[i]['text']) < short_length:
            # merge i with i+1 paragraph
            cur_p = para_list.pop(i)
            next_p = para_list.pop(i)
            merge_p = merge_two_paragraphs(cur_p, next_p)
            para_list.insert(i, merge_p)
            # check the merged paragraph in the next loop
        else:
            i += 1

    return para_list

if __name__ == '__main__':
    from pathlib import Path
    import json
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help = 'file of cleaned cuad data')
    parser.add_argument('output_path')
    parser.add_argument('--threshold', type = int, default = 300)
    args = parser.parse_args()

    with open(args.input_path) as f:
        ori_data = [json.loads(k) for k in f]
    
    for sample in ori_data:
        sample['paras'] = merge_paragraphs_and_relocate(
            sample['paras'], short_length = args.threshold
        )
    
    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents = True, exist_ok = True)

    with open(out_p, 'w') as f:
        for d in ori_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')