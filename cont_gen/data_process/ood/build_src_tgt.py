"""Build the supervised source and target data"""

from typing import Any
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

from cont_gen.data_process.utils import rand_choice, ommit_middle
from cont_gen.utils import load_json, load_jsonl, save_jsonl

def cut_to_max_length(text, tokenizer, max_len):
    """Cut a text so that the length of its tokens do not exceed max_len"""
    token_ids = tokenizer.encode(text, add_special_tokens = False)
    dec_text: str = tokenizer.decode(token_ids[:max_len])
    return dec_text.rstrip(chr(65533))

class Cutter:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __call__(self, text) -> str:
        return cut_to_max_length(text, self.tokenizer, self.max_len)
    
class NoCutter:
    """Do not cut"""
    def __call__(self, text) -> str:
        return text

class SFT_Builder:
    """Build the source and target text for SFT."""
    def __init__(self, prompt, clause_info: pd.DataFrame, all_para_data, cut_func):
        self.prompt = prompt
        self.clause_info = [row for _, row in clause_info.iterrows()]
        self.all_para_data = all_para_data
        self.title2para_text = {
            cont['title']: [para_d['text'] for para_d in cont['paras']] 
                for cont in all_para_data}
        self.cut_func = cut_func
    
    def __call__(self, meta_data: dict):
        clause_text = self.title2para_text[meta_data['title']][meta_data['para_idx']]
        cinfo = self.clause_info[meta_data['q_id']]

        clause_text = self.cut_func(clause_text) # cut to max length

        source = self.build_prompt_src(self.prompt, 
                                              clause_text, 
                                              cinfo['clause_type'], 
                                              cinfo['clause_quest'])
        target = self.build_prompt_tgt(meta_data['answers'])

        return source, target
    
    @staticmethod
    def build_prompt_src(prompt, text, clause_type, clause_quest):
        return prompt.format(text = text, clause_type = clause_type, clause_quest = clause_quest)
    
    @staticmethod
    def build_prompt_tgt(answers):
        if len(answers) > 0:
            return '\n'.join(['- ' + a for a in answers])
        else:
            return 'No'

class SFT_Builder_YesNo(SFT_Builder):
    @staticmethod
    def build_prompt_tgt(answers):
        if len(answers) > 0:
            tgt = '\n'.join(['- ' + a for a in answers])
            tgt = 'Yes. ' + tgt
            return tgt
        else:
            return 'No.'

class SFT_Builder_YesNo_Natural(SFT_Builder):
    @staticmethod
    def build_prompt_tgt(answers):
        if len(answers) > 0:
            tgt = '\n'.join(['- ' + a for a in answers])
            tgt = 'There are such clauses. They are:\n' + tgt
            return tgt
        else:
            return 'There is no such clause.'

def process(builder:SFT_Builder, meta_df:pd.DataFrame):
    all_data = []
    for _, row in tqdm(list(meta_df.iterrows())):
        row = row.to_dict()
        source, target = builder(row)
        pro_data = {
            'title': row['title'],
            'para_idx': row['para_idx'],
            'q_id': row['q_id'],
            'source': source,
            'target': target
        }
        # Here, we use 'type' to indicate different data split.
        if 'small' in row:
            pro_data['type'] = row['small']
        if 'type' in row:
            pro_data['type'] = row['type']
        all_data.append(pro_data)
    return all_data

def main():
    import argparse
    from pathlib import Path
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('split_dir', help = 'dir that holds train and test meta data')
    parser.add_argument('--type', default = 'basic', help = 'basic, yes_no, yes_no_natural')
    parser.add_argument('--prompt', help = 'prompt path')
    parser.add_argument('--save_dir', help = 'path to save source and target data.')
    parser.add_argument('--tokenizer', default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--max_len', type = int, default = 512)

    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    # get default save_dir, which is under split_dir and named with the prompt
    if args.save_dir is None:
        save_dir = split_dir / Path(args.prompt).stem
        print(f'Save to {str(save_dir)}')
    else:
        save_dir = Path(args.save_dir)
    
    prompt = open(args.prompt, 'r').read()
    print(f'Prompt:\n{prompt}')

    clause_info = pd.read_csv('./data/clause/all_info.csv')
    all_para_data = load_jsonl('./data/cuad_clean/CUADv1_paras_merge_new.jsonl')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cutter = Cutter(tokenizer, args.max_len)

    if args.type == 'basic':
        builder_cls = SFT_Builder
    elif args.type == 'yes_no':
        builder_cls = SFT_Builder_YesNo
    elif args.type == 'yes_no_natural':
        builder_cls = SFT_Builder_YesNo_Natural

    builder = builder_cls(prompt, clause_info, all_para_data, cutter)

    train_meta = pd.read_csv(str(split_dir / 'train_meta.csv'), converters={'answers': literal_eval})
    train_data  = process(builder, train_meta)
    save_jsonl(train_data, save_dir / 'train_data.jsonl')

    test_meta_id = pd.read_csv(str(split_dir / 'test_meta_id.csv'), converters={'answers': literal_eval})
    test_data_id  = process(builder, test_meta_id)
    save_jsonl(test_data_id, save_dir / 'test_data_id.jsonl')

    test_meta_ood = pd.read_csv(str(split_dir / 'test_meta_ood.csv'), converters={'answers': literal_eval})
    test_data_ood  = process(builder, test_meta_ood)
    save_jsonl(test_data_ood, save_dir / 'test_data_ood.jsonl')


if __name__ == '__main__':
    main()