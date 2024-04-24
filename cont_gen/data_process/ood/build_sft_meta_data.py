"""
Given training and test label sets, build SFT meta data with negative sampling.

No prompt is included here.

Data format:
    source_text,
    q_id,
    target_text
"""

import random
import numpy as np
from collections import Counter, OrderedDict, defaultdict
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

from cont_gen.data_process.utils import rand_choice, ommit_middle
from cont_gen.utils import load_json, load_jsonl, save_jsonl

class CUAD_Basic:
    """
    Hold basic information of cuad data
    """
    def __init__(self, 
                 clause_info_path, 
                 para_data_path, 
                 train_title_path, 
                 test_title_path):
        self.clause_info = pd.read_csv(clause_info_path)
        self.para_data = load_jsonl(para_data_path)
        self.train_titles = load_json(train_title_path)
        self.test_titles = load_json(test_title_path)

        # title2para = {k['title']: k['paras'] for k in self.para_data}
        self.train_para_data = [k for k in self.para_data if k['title'] in self.train_titles]
        self.test_para_data = [k for k in self.para_data if k['title'] in self.test_titles]


class MetaSFT_Train_Builder:
    meta_columns = ['title', 'para_idx', 'q_id', 'answers']

    @staticmethod
    def build_pos_neg_samples(para_data,
                              label_ids, 
                              neg_clause_ratio: float = 1.0, 
                              num_neg_quest: int = 1):
        pos_df = MetaSFT_Train_Builder.get_pos(para_data, label_ids)

        pos_df.insert(4, 'type', [0] * len(pos_df))

        # quest 2 clause neg
        neg_q2c_df = MetaSFT_Train_Builder.get_neg_for_quest(
            para_data, pos_df, neg_clause_ratio
        )
        neg_q2c_df.insert(4, 'type', [1] * len(neg_q2c_df))

        # clause 2 quest neg
        neg_c2q_df = MetaSFT_Train_Builder.get_neg_for_clause(pos_df, label_ids, num_neg_quest)
        neg_c2q_df.insert(4, 'type', [2] * len(neg_c2q_df))

        total_df = pd.concat([pos_df, neg_q2c_df, neg_c2q_df], axis = 0)

        return total_df.reset_index()
    
    @staticmethod
    def get_pos(total_para_data, label_ids)->pd.DataFrame:
        pos_samples = []
        for para_d in total_para_data:
            for pi, para in enumerate(para_d['paras']):
                for qa in para['qas']:
                    if qa['q_id'] not in label_ids:
                        continue
                    aws_list = [k['text'] for k in qa['answers']]
                    assert len(aws_list) > 0
                    pos_samples.append([
                        para_d['title'], pi, qa['q_id'], aws_list])
        return pd.DataFrame(pos_samples, columns = MetaSFT_Train_Builder.meta_columns)
    
    @staticmethod
    def get_neg_for_quest(total_para_data, pos_df: pd.DataFrame, neg_c_ratio):
        """For each question, sample negative clause"""
        q2num_pos = pos_df['q_id'].value_counts().to_dict()
        q2num_neg = {k: int(v * neg_c_ratio) for k,v in q2num_pos.items()}
        return MetaSFT_Train_Builder.get_neg_for_quest_inner(
            total_para_data, q2num_neg
        )

    @staticmethod
    def get_neg_for_quest_inner(total_para_data, label_id2num: dict) -> pd.DataFrame:
        """
        For each question, get the number of negative samples specified in label_id2num.
        """
        neg_samples = []
        for q_id, num_neg in label_id2num.items():
            cla_neg = [] # (title, para_idx, q_id, text, answers)
            while len(cla_neg) < num_neg:
                # choose a random contract
                cont_idx = random.choice(range(len(total_para_data)))
                cont_data = total_para_data[cont_idx]
                
                # choose a random paragraph
                para_idx = random.choice(range(len(cont_data['paras'])))
                para = cont_data['paras'][para_idx]
                
                # make sure the paragraph does not contain the clause
                if q_id in [qa['q_id'] for qa in para['qas']]:
                    continue
                
                # make sure this neg sample has not been selected.
                neg_sp_meta = (cont_data['title'], para_idx, q_id)
                if neg_sp_meta in cla_neg:
                    continue
                cla_neg.append(neg_sp_meta)
                
                # add this negative sample
                neg_samples.append([*neg_sp_meta, []])
        return pd.DataFrame(neg_samples, columns = MetaSFT_Train_Builder.meta_columns)
    
    @staticmethod
    def get_neg_for_clause(pos_df: pd.DataFrame, label_ids, n_neg_q: int):
        """For each clause, sample negative questions."""
        # First, get the mapping from (title, para_idx) to q_ids
        p2qs = defaultdict(list)
        for _, row in pos_df.iterrows():
            p_uid = (row['title'], row['para_idx'])
            p2qs[p_uid].append(row['q_id'])

        # sample neg questions for each clause
        neg_samples = []
        for (title, p_i), qids in p2qs.items():
            neg_ids = [k for k in label_ids if k not in qids]
            # shuffle
            random.shuffle(neg_ids)
            for q_i in neg_ids[:n_neg_q]:
                neg_samples.append([title, p_i, q_i, []])
        
        return pd.DataFrame(neg_samples, columns = MetaSFT_Train_Builder.meta_columns)

class MetaSFT_Test_Builder:
    meta_columns = ['title', 'para_idx', 'q_id', 'answers', 'small']

    @staticmethod
    def build_test_and_small(total_para_data, label_ids, neg_ratio: float = 0.1):
        """
        Build the full test set (num_para * num_quest) and a small subset for fast evaluation.
        """
        all_samples = []
        for para_d in total_para_data:
            num_para = len(para_d['paras'])
            # Create a 2-d matrix, row refers to paragraphs, column refer to questions
            mat = np.zeros((num_para, len(label_ids)), dtype = np.int64)
            # First, for positive samples, mark them as 1
            for pi, para in enumerate(para_d['paras']):
                for qa in para['qas']:
                    if qa['q_id'] in label_ids:
                        q_col = label_ids.index(qa['q_id'])
                        mat[pi, q_col] = 1
            # Then, if one paragraph has positive clause types, mark the row to 2
            for pi in range(mat.shape[0]):
                if any(mat[pi] == 1):
                    mat[pi][mat[pi] == 0] = 2
            # Last, for each clause(column), sample neg paragraphs and mark as 3
            for q_col in range(mat.shape[1]):
                # sample neg_ratio * n_para negative samples
                col = mat[:, q_col]
                neg_idx = np.where(col == 0)[0]
                n_neg = len(col) * neg_ratio
                np.random.shuffle(neg_idx)
                sel_neg_idx = neg_idx[:int(n_neg)]
                mat[sel_neg_idx, q_col] = 3
            
            for pi in range(mat.shape[0]):
                for q_col in range(mat.shape[1]):
                    q_i = label_ids[q_col]
                    small = mat[pi, q_col]
                    if small == 1:
                        qas = para_d['paras'][pi]['qas']
                        qa = [k for k in qas if k['q_id'] == q_i][0]
                        answers = [k['text'] for k in qa['answers']]
                    else:
                        answers = []
                    all_samples.append([
                        para_d['title'], pi, q_i, answers, small
                    ])
        return pd.DataFrame(all_samples, columns= MetaSFT_Test_Builder.meta_columns)
     

class OOD_Builder:
    def __init__(self, cuad_basic: CUAD_Basic, rand_seed, output_dir, num_train_labels = 29):
        self.cuad_basic = cuad_basic
        n_tot_lab = len(self.cuad_basic.clause_info)
        # randomly split train and test labels
        x = list(range(n_tot_lab))
        random.Random(rand_seed).shuffle(x)

        self.train_labels = sorted(x[:num_train_labels])
        self.test_labels = sorted(x[num_train_labels:])

        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def save_labels(self):
        # save label list
        out_dir = Path(self.output_dir)

        id2label = {i:row['clause_type'] for i, row in self.cuad_basic.clause_info.iterrows()}
        tr_lab_df = pd.DataFrame({
            'clause_id': self.train_labels,
            'clause_type': [id2label[k] for k in self.train_labels]
        })
        te_lab_df = pd.DataFrame({
            'clause_id': self.test_labels,
            'clause_type': [id2label[k] for k in self.test_labels]
        })

        tr_lab_df.to_csv(out_dir / 'train_labels.csv', index = False)
        te_lab_df.to_csv(out_dir / 'test_labels.csv', index = False)

    def build_train_meta(self, neg_clause_ratio=1.0, num_neg_quest = 1):
        all_df = MetaSFT_Train_Builder.build_pos_neg_samples(
            self.cuad_basic.train_para_data,
            self.train_labels,
            neg_clause_ratio=neg_clause_ratio,
            num_neg_quest=num_neg_quest)
        
        stat = all_df['type'].value_counts()
        print(stat)

        all_df.to_csv(Path(self.output_dir) / 'train_meta.csv', index = False)

        return all_df

    def build_test_meta(self):
        """
        All questions. add a column `small` to indicate the small subset for fast evaluation.
        """
        tot_para_data = self.cuad_basic.test_para_data

        # OOD test
        test_df = MetaSFT_Test_Builder.build_test_and_small(tot_para_data, self.test_labels, neg_ratio = 0.1)

        print(f'Total test: {len(test_df)}. Small types:')
        stat = test_df['small'].value_counts()
        print(stat)
        
        test_df.to_csv(Path(self.output_dir) / 'test_meta_ood.csv', index = False)

        # ID test
        test_id_df = MetaSFT_Test_Builder.build_test_and_small(tot_para_data, self.train_labels, neg_ratio = 0.1)

        # print(f'Total test (ID): {len(test_id_df)}. Small types:')
        # stat = test_id_df['small'].value_counts()
        # print(stat)
        
        # test_id_df.to_csv(Path(self.output_dir) / 'test_meta_id.csv', index = False)

        return test_df, test_id_df
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type = int)
    args = parser.parse_args()

    proj_dir = Path('./')
    cuad_basic = CUAD_Basic(
        proj_dir / 'data/clause/all_info.csv',
        proj_dir / 'data/cuad_clean/CUADv1_paras_merge_new.jsonl',
        proj_dir / 'data/cuad_split/ori_train_titles.json',
        proj_dir / 'data/cuad_split/ori_test_titles.json',
    )

    n_tr_lab = 29
    builder = OOD_Builder(cuad_basic, args.seed, 
                          proj_dir / f'data/ood_split/seed{args.seed}_tr{n_tr_lab}',
                          num_train_labels=n_tr_lab)

    print('save labels')
    builder.save_labels()

    builder.build_train_meta()

    builder.build_test_meta()