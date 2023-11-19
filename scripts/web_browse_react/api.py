import os
import sys
import json
from flask import Flask, request, jsonify, make_response
from dotenv.main import dotenv_values
from pathlib import Path

def get_web_basic_app(app = None):
    """Return a flask app with routes to home page and css js files"""
    if app is None:
        app = Flask(__name__)

    @app.route('/')
    def homepage():
        with open('src/index.html', encoding='utf8') as f:
            page = f.read()
        return page

    @app.route('/<fn>')
    def get_web(fn):
        with open('./src/{}'.format(fn), encoding='utf8') as f:
            s = f.read()
        r = make_response(s)
        if fn.split('.')[-1] == 'css':
            r.mimetype = 'text/css'
        elif fn.split('.')[-1] in ['js']:
            r.mimetype = 'application/javascript'
        elif fn.split('.')[-1] in ['jsx']:
            r.mimetype = 'application/x-javascript+xml'
            
        return r
    
    return app


class CUAD_Data:
    """
    Handle CUAD data.

    Args:
        path: path of the squad format data
    
    Support:
        Contract search by fuzzy matching of title
        return different spans given a contract and title
    """
    def __init__(self, path = None):
        if path is  None:
            ENVS = dotenv_values('../../.env')
            path = Path(ENVS['CUAD_PATH']) / 'CUAD_v1.json'
        
        self.data = json.load(open(path))['data']
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
        self.history = [] # record browse history
        self.clause_info = json.load(open('../../data/clause_info.json'))


    def get_contract(self, idx):
        """Return a dict of contract title and text"""
        cont_data = self.data[idx]
        self.update_history(idx)
        return {
            'title': cont_data['title'], 
            'contract_text': cont_data['paragraphs'][0]['context']
        }

    def update_history(self, idx):
        if idx in self.history:
            # remove the current contract in the history
            pos = self.history.index(idx)
            _ = self.history.pop(pos)
        self.history.append(idx)
    
    def get_history(self):
        """Return history index and title"""
        return [{'index': k,'title': self.data[k]['title']} for k in self.history]

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
    

def main():
    # app = Flask(
    #     __name__,
    #     static_url_path='/',
    #     static_folder = './src'
    # )
    app = get_web_basic_app()

    # handler = CUAD_Data()

    # app = get_data_app(handler, app)

    app.run(host = '0.0.0.0', port = 3000, threaded = True)

if __name__ == '__main__':
    main()