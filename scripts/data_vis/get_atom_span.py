"""
Given a document and highlighted spans of different types, 
return the atom spans whose tokens have same combined type.
"""

from typing import List, Dict, Any, Tuple

def get_atom_span(doc: str, all_spans: Dict[str, List[Tuple[int, int]]], add_text = True):
    """
    For each span type, there may exist multiple spans.
    Args:
        doc: document to split into atom spans
        all_spans: map from span type to a list of spans,
            each span is a tuple of  [start, end]
            end is not included
    Return:
        atom_spans: a list of atom_span, which is a dict:
            start: start position
            end: end position
            text: span text from doc. exists if add_text = True
            types: a list of span types
    """
    print(all_spans)
    # 1. Get token types
    token_type_map = [set() for _ in doc]

    for span_type, spans in all_spans.items():
        for start, end in spans:
            end = min(end, len(doc))
            for i in range(start, end):
                token_type_map[i].add(span_type)
    
    # 2. Get atom spans by walking through tokens
    atom_spans = []
    cur_atom_span = None
    for i, token_types in enumerate(token_type_map):
        if cur_atom_span is None:
            # at begin of doc 
            cur_atom_span = {'start': 0, 'end': i+1, 'types': token_types}
        else:
            if token_types == cur_atom_span['types']:
                # continuation in atom_span
                cur_atom_span['end'] = i + 1
            else:
                # arrive at the begin of a new atom span
                atom_spans.append(cur_atom_span)
                cur_atom_span = {'start': i, 'end': i+1, 'types': token_types}
    atom_spans.append(cur_atom_span)

    # convert span type to list and add text
    for k in atom_spans:
        k['types'] = list(k['types'])
        if add_text:
            k['text'] = doc[k['start']: k['end']]
    
    return atom_spans

if __name__ == '__main__':
    doc = ''.join(map(str, range(10)))
    all_spans = {
        'person': [[3,7]],
        'loc': [[5,8]]
    }
    for k in get_atom_span(doc, all_spans):
        print(k)
