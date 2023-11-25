import os
import sys
import json
from flask import Flask, request, jsonify, make_response
from dotenv.main import dotenv_values
from pathlib import Path

from data_handler import CUAD_Data, get_data_app

def get_web_basic_app(app = None):
    """Return a flask app with routes to home page and css js files"""
    if app is None:
        app = Flask(__name__)

    @app.route('/')
    def homepage():
        with open('home.html', encoding='utf8') as f:
            page = f.read()
        return page

    # static route
    @app.route('/<fn>')
    def get_web(fn):
        with open('./web/{}'.format(fn), encoding='utf8') as f:
            s = f.read()
        r = make_response(s)
        if fn.split('.')[-1] == 'css':
            r.mimetype = 'text/css'
        elif fn.split('.')[-1] == 'js':
            r.mimetype = 'application/javascript'
        return r
    
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--port', type = int, default = 3000)
    args = parser.parse_args()

    ENVS = dotenv_values('../../.env')
    

    app = get_web_basic_app()

    if args.test:
        path = Path(ENVS['CUAD_TEST'])
        pred_path = '../../data/test_pred_spans.pkl'
    else:
        path = Path(ENVS['CUAD_PATH']) / 'CUAD_v1.json'
        pred_path = None
    
    handler = CUAD_Data(path, pred_path)

    app = get_data_app(handler, app)

    app.run(host = '0.0.0.0', port = args.port, threaded = True)

if __name__ == '__main__':
    main()