from cont_gen.data_process.utils import overlap_of_two_span, chunk_text
from cont_gen.utils import save_jsonl

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

if __name__ == '__main__':
    from pathlib import Path
    import json
    import argparse
    from tqdm import tqdm
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help = 'file of paragraph data')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer')
    parser.add_argument('max_para_len', type = int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code = True)

    with open(args.input_path) as f:
        ori_data = [json.loads(k) for k in f]
    
    pro_data = [
        split_para_to_chunk_para(d, tokenizer, args.max_para_len) 
            for d in tqdm(ori_data, ncols = 80)]
    
    # create output dir
    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents = True, exist_ok = True)

    # save data
    save_jsonl(pro_data, out_p)
