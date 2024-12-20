from dotenv.main import dotenv_values
from pathlib import Path
from transformers import AutoTokenizer
import os

from . import context
from cont_gen.data_process.squad_balance import CUAD_Dataset

def main():
    ENVS = dotenv_values('./.env')
    print(ENVS)
    cuad_dir = Path(ENVS['CUAD_PATH'])
    tk = AutoTokenizer.from_pretrained('roberta-base', use_fast = True)
    ds = CUAD_Dataset(
        cuad_dir / 'CUAD_v1.json', tk, 'train',
        doc_stride=256,
        max_query_length = 128,
        max_seq_length=512,
        num_cpu = 2
    )
    print(ds[0].keys())

if __name__ == '__main__':
    main()