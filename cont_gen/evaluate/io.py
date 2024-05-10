"""
Utilities to load prediction results and metrics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from IPython.display import display

from cont_gen.utils import load_jsonl, get_ckpt_paths, save_json
from cont_gen.evaluate.cal_metrics import get_overall_metrics

class RunManager:
    """
    Attributes:
        runs: (pd.DataFrame) with fields:
            model, split, run_name, has_pred, has_metrics
                has_pred: 0: no, 1: sampeld, 2: all
        ckpt_details: (pd.DataFrame) with fields
            run_path, ckpt, pred_all, pred_sampled, metric_all, metric_sampled
    """
    def __init__(self, top_dir, is_ood = True):
        self.top_dir = top_dir
        self.is_ood = is_ood
        self.dist = 'ood' if is_ood else 'id'

        self.init_tables()

        self.scan_top()
    
    def init_tables(self):
        self.runs = pd.DataFrame(columns = ['model', 'split', 'run_name',
                                             'has_pred', 'has_metric'])
        self.ckpt_details = pd.DataFrame(columns=[
            'run_path', 'ckpt', 'pred_all', 'pred_sampled', 'metric_all', 'metric_sampled'
        ])

    def scan_top(self):
        """Scan all folders and runs"""
        self.init_tables()
        top_dir = Path(self.top_dir)
        for model_path in top_dir.glob('*/'):
            if not model_path.is_dir():
                continue
            for split_path in model_path.glob('*/'):
                if not split_path.is_dir():
                    continue
                for run_path in split_path.glob('*/'):
                    if not run_path.is_dir():
                        continue
                    self.scan_run(run_path)
    
    def scan_run(self, run_path):
        """Scan one run"""
        # resolve path
        run_path = Path(run_path)
        model, split, run_name = run_path.parts[-3:]
        ckpts = get_ckpt_paths(run_path)

        # scan checkpoints
        for ckpt in ckpts:
            res = self.find_ckpt(ckpt)
            self.add_df_row(self.ckpt_details, [str(run_path), ckpt.name, *res])
        
        # summarize run results
        part = self.ckpt_details[self.ckpt_details['run_path'] == str(run_path)]
        has_pred_all = any(k is not None for k in part['pred_all'])
        has_pred_sampled = any(k is not None for k in part['pred_sampled'])
        has_pred = 2 if has_pred_all else 1 if has_pred_sampled else 0

        has_metric_all = any(k is not None for k in part['metric_all'])
        has_metric_sampled = any(k is not None for k in part['metric_sampled'])
        has_metric = 2 if has_metric_all else 1 if has_metric_sampled else 0

        self.add_df_row(self.runs, [model, split, run_name, has_pred, has_metric])
            
    
    def find_ckpt(self, ckpt) -> List[Optional[Path]]:
        """Find prediction and metric files under ckpt folder"""
        ckpt = Path(ckpt)
        pred_all = self.exist(ckpt / f'predictions_{self.dist}_all.jsonl')
        pred_sampled = self.exist(ckpt / f'predictions_{self.dist}_sampled.jsonl')
        metric_all = self.exist(ckpt / f'detail_metrics_{self.dist}_all.csv')
        metric_sampled = self.exist(ckpt / f'detail_metrics_{self.dist}_sampled.csv')
        return pred_all, pred_sampled, metric_all, metric_sampled
    
    @staticmethod
    def exist(path) -> Optional[Path]:
        if Path(path).exists():
            return path
        else:
            return None

    @staticmethod
    def add_df_row(df, row):
        assert len(df.columns) == len(row), f'{len(df.columns)} != {len(row)}'
        df.loc[len(df)] = row


    def print_no_pred_runs(self):
        part = self.runs[self.runs['has_pred'] == 0]
        display(part)
    
    def cal_metrics_for_ckpt(self, ckpt, ground_df, parse_fn):
        ckpt = Path(ckpt)
        if not ckpt.exists():
            return

        pred_all = self.exist(ckpt / f'predictions_{self.dist}_all.jsonl')
        pred_sampled = self.exist(ckpt / f'predictions_{self.dist}_sampled.jsonl')

        ov_all_path = ckpt / f'ov_metrics_{self.dist}_all.csv'
        ov_sampled_path = ckpt / f'ov_metrics_{self.dist}_sampled.csv'


        if pred_all is not None:
            if ov_all_path.exists():
                return
            pred_all_df = pd.DataFrame(load_jsonl(pred_all))
            pred_sampled_df = pred_all_df[pred_all_df['type'] > 0]

            # handle all metrics
            ov_mets_all, detail_df_all = get_overall_metrics(ground_df, pred_all_df, parse_fn)
            save_json(ov_mets_all, ov_all_path)
            detail_df_all.to_csv(ckpt / f'detail_metrics_{self.dist}_all.csv', index = False)

            # handle sampled metrics
            ov_mets_sampled, detail_df_sampled = get_overall_metrics(ground_df, pred_sampled_df, parse_fn)
            save_json(ov_mets_sampled, ov_sampled_path)
            detail_df_sampled.to_csv(ckpt / f'detail_metrics_{self.dist}_sampled.csv', index = False)
            
        elif pred_sampled is not None:
            if ov_sampled_path.exists():
                return
            pred_sampled_df = pd.DataFrame(load_jsonl(pred_sampled))
            ov_mets_sampled, detail_df_sampled = get_overall_metrics(ground_df, pred_sampled_df, parse_fn)
            save_json(ov_mets_sampled, ov_sampled_path)
            detail_df_sampled.to_csv(ckpt / f'detail_metrics_{self.dist}_sampled.csv', index = False)
        
        else:
            return
    
    def cal_metrics_for_run(self, run_path, ground_df, parse_fn):
        ckpts = get_ckpt_paths(run_path)
        for ckpt in ckpts:
            print(f'Calcualte for {ckpt.name}')
            self.cal_metrics_for_ckpt(ckpt, ground_df, parse_fn)