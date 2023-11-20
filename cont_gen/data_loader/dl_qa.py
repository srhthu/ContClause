"""Dataloader for qa model"""
import torch
from typing import List, Dict, Any
class Data_Collator:
    columns = None

    def stack_columns(self, features: List[Dict[str, Any]], columns: List[str]):
        new_feats = {
            col:torch.stack([torch.tensor(feat[col]) for feat in features], dim = 0)
              for col in columns
        }
        return new_feats
    
    def __call__(self, features):
        columns = self.columns if self.columns else list(features[0].keys())
        return self.stack_columns(features, columns)

class CUAD_QA_Collator(Data_Collator):
    columns = [
        'input_ids', 'attention_mask', 'token_type_ids', 
        'start_position', 'end_position'
    ]
    def __call__(self, features):
        features = self.stack_columns(features, self.columns)
        col_map = {
            'start_position': 'start_positions', 
            'end_position':'end_positions'
        }
        for old_k, new_k in col_map.items():
            v = features.pop(old_k)
            features[new_k] = v
        return features
