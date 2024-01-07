"""Shared functions"""
from functools import cmp_to_key
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import random

def convert_token_char_map(token_to_char: List[Tuple[int, int]], length):
    """
    Given the token_to_char map, return the char_to_token map.
    For space that not in token, return None.
    Both map is a mapping to span position.

    Note: one char can also be mapped to several tokens.
    """
    char_to_token = [[None, None] for _ in range(length)]
    for i, (start, end) in enumerate(token_to_char):
        for j in range(start, end):
            if char_to_token[j][0] is None:
                # start token of the j-th char
                char_to_token[j][0] = i 
            char_to_token[j][1] = i + 1
    return char_to_token

def reverse_char_map(char_map, tot_len = None):
    """
    Given the char_map from clean_text to ori_text, get the reverse one.

    Note: the reverse is a n-1 mapping
    """
    if tot_len is None:
        tot_len = char_map[-1] + 1
    r_char_map = [None for _ in range(tot_len)]

    for i, ori_i in enumerate(char_map):
        r_char_map[ori_i] = i
    
    while i < len(r_char_map):
        if r_char_map[i] is None:
            r_char_map[i] = 0
            i += 1
        else:
            if i+1 < len(r_char_map) and r_char_map[i+1] is None:
                # brodcast current i to next one
                r_char_map[i+1] = r_char_map[i]
            i += 1
    return r_char_map

# Functions related to remove some spans from a string
def cut_spans_return_offset(ori_text: str, span_pos: List[Tuple[int, int]]):
    """
    Cut the ori_text with specified spans and return the offset of each part.

    The end position is NOT included.

    Return:
        text_parts (List[str])
        offsets (List[int])
    """
    # sort the spans
    span_pos = sorted(span_pos, key = lambda k: k[0])
    
    # add a pseudo span at the end of the doc
    if len(span_pos) == 0 or span_pos[-1][-1] < len(ori_text):
        # add a pseudo linebreak at the end of the doc
        span_pos.append([len(ori_text), None])    
    
    # get split text parts
    text_parts = []
    offsets = []
    cur_start = 0 # the start position of current paragraph
    # traverse the removed spans and add the previous text trunk into text_parts
    for sp_st, sp_end in span_pos:
        p_text = ori_text[cur_start: sp_st]
        if len(p_text) > 0:
            # make sure the removed span is NOT at the begin of ori_text
            text_parts.append(p_text)
            offsets.append(cur_start)

        cur_start = sp_end
    return text_parts, offsets

def remove_spans_and_return_mapping(ori_text: str, span_pos: List[Tuple[int, int]]):
    """
    Remove text's spans with position specified in span_pos, and 
    return the position map from the new_text to ori_text.

    The end position is NOT included.

    Return:
        new_text (str)
        new_to_old_mapping (List[int])
    """

    text_parts, offsets = cut_spans_return_offset(ori_text, span_pos)

    new_to_old_mapping = []
    for t, ofs in zip(text_parts, offsets):
        new_to_old_mapping.extend([ofs + i for i in range(len(t))])
    
    return ''.join(text_parts), new_to_old_mapping


# Functions related to span spatial relations
def cmp_span(span1, span2):
    if span1[0] < span2[0]:
        return -1
    elif span1[0] > span2[0]:
        return 1
    else:
        if span1[1] < span2[1]:
            return -1
        elif span1[1] > span2[1]:
            return 1
        else:
            return 0

def overlap_of_two_span(pos_1, pos_2):
    """
    Given the position of two span, return the overlap span position if any or None.
    """
    # if there are overlapping, only two case
    # the left of one span fall into the other span

    # We use a heuristic method to determin the overlapping boundary
    ol_left = max(pos_1[0], pos_2[0])
    ol_right = min(pos_1[1], pos_2[1])

    if ol_left <= ol_right:
        return (ol_left, ol_right)
    else:
        return None
    

def merge_spans(span_pos: List[Tuple[int, int]]):
    """
    Merge overlapped spans and return in order.
    """
    sorted_spans = sorted(span_pos, key = cmp_to_key(cmp_span))
    i = 0
    while i < len(sorted_spans) - 1:
        cur_span = sorted_spans[i]
        next_span = sorted_spans[i+1]
        if cur_span[0] == next_span[0]:
            if cur_span[1] == next_span[1]:
                # duplicate span
                _ = sorted_spans.pop(i + 1)
                continue
            else:
                _ = sorted_spans.pop(i)
                continue
        else:
            if next_span[0] <= cur_span[1]:
                # merge two span
                m_span = [cur_span[0], max(cur_span[1], next_span[1])]
                _ = sorted_spans.pop(i)
                _ = sorted_spans.pop(i)
                sorted_spans.insert(i, m_span)
                continue
            else:
                # no overlapping
                i += 1
    return sorted_spans

def span_contain(span1, span2):
    """Return whether span1 is included in span2"""
    if span1[0] >= span2[0] and span1[1] <= span2[1]:
        return True
    return False

def relocate_spans_into_range(spans, start, end):
    """Filter out spans outside (start, end) and relocate the position"""
    for span in spans:
        if span_contain(span, (start, end)):
            yield [span[0] - start, span[1] - start]


# Functions related to CUAD dataset
def get_doc_with_spans(data):
    """
    Return:
        sim_dataset: a list of examples:
            title: contract title
            doc: contract content
            all_answers: a list of answers for each question:
                answers: a list of spans (Tuple[int, int])
    """
    sim_dataset = []
    for exa in data:
        title = exa['title']
        pqa = exa['paragraphs'][0]
        doc = pqa['context']
        all_answers = []
        for qa in pqa['qas']:
            anss = qa['answers']
            spans = [[k['answer_start'], k['answer_start'] + len(k['text'])] for k in anss]
            all_answers.append(spans)
        sim_dataset.append({
            'title': title,
            'doc': doc,
            'all_answers': all_answers
        })
    return sim_dataset

# Others
def group_by(examples: List[Dict[str, Any]], key):
    """
    Groub examples by key. Follow the group presence order.
    """
    groups = []
    gname2id = {} # group identifier to index
    for e in examples:
        gname = e[key]
        if gname not in gname2id:
            gname2id[gname] = len(groups)
            groups.append([e])
        else:
            groups[gname2id[gname]].append(e)
    return groups

def rand_choice(l, n):
    """Randomly choice n elements of list l without replacement"""
    if n > len(l):
        return [*l]
    return random.sample(l, n)