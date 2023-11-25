import json
from pathlib import Path
import sys
from flask import Flask, request, jsonify, make_response
from dotenv.main import dotenv_values
from typing import List
import pickle
from typing import List, Dict, Tuple

sys.path.insert(0, '../data/')
from get_atom_span import get_atom_span

def get_cuad_atom_span(data, pred_spans = {}, add_text = True):
    para_data = data['paragraphs'][0]
    doc = para_data['context']
    all_spans = {}
    for i, qa in enumerate(para_data['qas']):
        spans = [
            (
                k['answer_start'], 
                k['answer_start'] + len(k['text'])
            ) 
            for k in qa['answers']
        ]
        all_spans[f'gold_{i}'] = spans
    all_spans.update(pred_spans)
    return get_atom_span(doc, all_spans, add_text= add_text)

class CUAD_Data:
    """
    Handle CUAD data.

    Args:
        path: path of the squad format data
    
    Support:
        Contract search by fuzzy matching of title
        return different spans given a contract and title
    """
    def __init__(self, path = None, pred_path = None):
        if path is  None:
            ENVS = dotenv_values('../../.env')
            path = Path(ENVS['CUAD_PATH']) / 'CUAD_v1.json'
        
        self.data = json.load(open(path))['data']
        if pred_path is not None:
            self.pred_spans:List[Dict[str, List[Tuple[int, int]]]] = pickle.load(open(pred_path, 'rb'))
            
        else:
            self.pred_spans = [{} for _ in self.data]
        self.atom_span_cache = [None for _ in self.data]

        """
        title: contract id
        paragraphs: a list of length 0
            context
            qas: a list of question answer dict:
                answers: a list of answer text and position dict:
                    text: text span in original contract
                    answer_start: int
                id: <title>__<clause type>
                question: a template contain the clause type
                is_impossible: True if there is no answer
        """
        self.history:List[int] = [] # record browse history
        self.clause_info = json.load(open('../../data/clause_info.json'))


    def get_contract(self, idx):
        """Return a dict of contract title and text"""
        idx = int(idx)
        if idx == -1:
            idx = 0 if len(self.history) == 0 else self.history[-1]
        idx = min(idx, len(self.data) - 1)
        idx = max(0, idx)
        cont_data = self.data[idx]
        self.update_history(idx)
        atom_spans = self.atom_span_cache[idx]
        if atom_spans is None:
            atom_spans = get_cuad_atom_span(cont_data, self.pred_spans[idx], add_text = False)
            self.atom_span_cache[idx] = atom_spans
        return {
            'idx': idx,
            'title': cont_data['title'], 
            'contract_text': cont_data['paragraphs'][0]['context'],
            'atom_spans': atom_spans,
            'impossible': [k['is_impossible'] for k in cont_data['paragraphs'][0]['qas']]
        }

    def update_history(self, idx):
        if idx in self.history:
            # remove the current contract in the history
            pos = self.history.index(idx)
            _ = self.history.pop(pos)
        self.history.append(idx)
    
    def get_history(self):
        """Return history index and title"""
        return [{'index': k,'title': self.data[k]['title']} for k in self.history[::-1]]


def get_data_app(handler: CUAD_Data, app = None):
    if app is None:
        app = Flask(__name__)
    
    @app.route('/get-history', methods = ['GET'])
    def _get_history():
        return jsonify(handler.get_history())
    
    @app.route('/get-clauses', methods = ['GET'])
    def _get_clauses():
        return jsonify(handler.clause_info)
    
    @app.route('/get-contract', methods = ['POST'])
    def _get_contract():
        req_data = request.json
        return jsonify(handler.get_contract(req_data['index']))
    
    return app