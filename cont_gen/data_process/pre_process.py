import re
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from typing import List, Tuple, Dict, Optional, Union

from transformers import PreTrainedTokenizerFast, AutoTokenizer

from cont_gen.data_process.basic import NaturalParagraph
from cont_gen.data_process.utils import (
    cut_spans_return_offset, 
    remove_spans_and_return_mapping,
    merge_spans,
    span_contain,
    relocate_spans_into_range,
    reverse_char_map,
    group_by
)


def remove_extra_spaces_and_newlines(doc):
    """
    Match the pattern of more than 2 spaces and newlines.
    """
    # for more than 2 consecutive spaces/newlines, we keep the first one and remove the left
    pat_sp = r' ([ ]+)'
    pat_nl = r'\n([\n]+)'

    cut_spans = []
    for pat in [pat_sp, pat_nl]:
        for m in re.finditer(pat, doc):
            cut_spans.append(m.span(1))
    
    new_doc, new2old_map = remove_spans_and_return_mapping(doc, cut_spans)
    return new_doc, new2old_map

def convert_cuad_sample(sample):
    doc = sample['paragraphs'][0]['context']

    # First, remove extra space/newline and keep the mapping
    new_doc, new2old_map = remove_extra_spaces_and_newlines(doc)
    old2new_map = reverse_char_map(new2old_map, len(doc))

    new_data = {
        'title': sample['title'],
        'doc_text': new_doc,
        'qas': [],
        'new2old_map': new2old_map
    }

    # traverse each question
    for old_qa in sample['paragraphs'][0]['qas']:
        old_answers = old_qa['answers']
        old_spans = [(k['answer_start'], k['answer_start'] + len(k['text'])) for k in old_answers]
        # get the new answer spans
        new_spans = [
            [old2new_map[left], old2new_map[right-1]+1]
                for left, right in old_spans
        ] # use the position of end character
        # In ideal case, the new_spans should not contain None

        # Merge overlapping spans
        new_spans = merge_spans(new_spans)

        new_qa = {
            'qa_id': old_qa['id'],
            'question': old_qa['question'],
            'is_impossible': old_qa['is_impossible'],
            'answers': [
                {
                    'text': new_doc[left: right],
                    'start_pos': left,
                    'end_pos': right - 1,
                }
                for left, right in new_spans
            ]
        }
        new_data['qas'].append(new_qa)
    
    return new_data

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    with open(args.input_path) as f:
        ori_data = json.load(f)
    
    pro_data = [convert_cuad_sample(k) for k in ori_data['data']]

    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents = True, exist_ok = True)

    with open(out_p, 'w') as f:
        for d in pro_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')