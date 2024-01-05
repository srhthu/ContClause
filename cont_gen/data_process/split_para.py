"""
Split document into paragraphs, convert answer span position, only keep possible answers, and merge short paragraphs.
"""

import json
import re
from typing import List, Dict, Any, Optional

from cont_gen.data_process.utils import (
    cut_spans_return_offset, 
    remove_spans_and_return_mapping,
    merge_spans,
    span_contain,
    relocate_spans_into_range,
    reverse_char_map,
    group_by
)

def split_one_document(sample):
    """
    Sample is an instance of cleaned document
    """
    # First, find all newlines and their position
    pat = '\n'
    break_spans = [m.span() for m in re.finditer(pat, sample['doc_text'])]
    text_parts, start_indexes = cut_spans_return_offset(sample['doc_text'], break_spans)

    para_data = []
    for p_text, offset in zip(text_parts, start_indexes):
        # find availabel questions
        p_len = len(p_text)
        p_qas = []
        for q_idx, qas in enumerate(sample['qas']):
            quest_answers = []
            for ai, answer in enumerate(qas['answers']):
                start_p = answer['start_pos']
                end_p = answer['end_pos']
                # determin whether answer span fall into the paragraph
                if start_p >= offset and end_p < (offset + p_len):
                    quest_answers.append({
                        'text': answer['text'],
                        'start_pos': start_p - offset,
                        'end_pos': end_p - offset,
                        'span_i': ai
                    })
            if len(quest_answers) > 0:
                p_qas.append({
                    'qa_id': qas['qa_id'],
                    'q_id': q_idx,
                    'answers': quest_answers
                })
        para_data.append({
            'text': p_text,
            'offset': offset,
            'qas': p_qas
        })
    return {'title': sample['title'], 'paras': para_data}

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help = 'file of cleaned cuad data')
    parser.add_argument('output_path')
    args = parser.parse_args()

    with open(args.input_path) as f:
        ori_data = [json.loads(k) for k in f]
    
    pro_data = [split_one_document(k) for k in ori_data]

    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents = True, exist_ok = True)

    with open(out_p, 'w') as f:
        for d in pro_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')