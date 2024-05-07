"""
Infer on prompts and save outputs.

Load test prompt data, generate on input with greedy decoding, get the probability of predict None
"""
import json
from argparse import ArgumentParser
from typing import List
from pathlib import Path

from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, GenerationConfig, 
    GenerationMixin, PreTrainedModel, PreTrainedTokenizer
)
from accelerate import Accelerator

from cont_gen.utils.model_utils import load_hf_model_from_checkpoint
from cont_gen.data_loader.cuad_prompt import SFT_Padding
from cont_gen.data_loader.cuad_sft import CUAD_SFT_Cached, CUAD_SFT_Filter_Type
from cont_gen.trainer.train_only_accelerate import Predictor
from cont_gen.utils import save_jsonl

MAX_NEW_TOKENS = 512

def load_test_dataset(args, tokenizer: PreTrainedTokenizer, is_seq2seq, part):
    """
    Load the test set.

    Args:
        part:
            all: load all test data
            sampled: load type > 0
            left: load type == 0
    """

    judge_type_fn_map = {
        'all': lambda k: True,
        'sampled': lambda k: k>0,
        'left': lambda k: k == 0
    }

    dataset = CUAD_SFT_Filter_Type(
        args.data_path,
        tokenizer,
        is_seq2seq = is_seq2seq,
        cache_dir = Path(args.data_path).parent / 'cache',
        max_src_length = args.max_length,
        is_test = True,
        small = args.debug,
        judge_type_fn = judge_type_fn_map[part]
    )
    return dataset

class SimpleGenerator:
    """Generate on prompts and return genereted string"""
    def __init__(self, tokenizer, is_encoder_decoder):
        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder
        self.gen_config = GenerationConfig(
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample = False,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id
        )
    def __call__(self, model: PreTrainedModel, batch):
        input_length = 1 if self.is_encoder_decoder else batch['input_ids'].shape[1]
        outputs = model.generate(
            input_ids = batch['input_ids'], 
            attention_mask = batch['attention_mask'],
            generation_config = self.gen_config
        )
        generated_tokens = outputs[:, input_length:]
        return {'pred_tokens': list(generated_tokens)}
    
    
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--data_path', help = 'path of test prompt data')
    parser.add_argument('--part', default = 'sampled',
                        help = 'which part to load based on type')
    parser.add_argument('--save_path')
    parser.add_argument('--base_model')
    parser.add_argument('--ckpt_dir', help = 'directory of model checkpoint')
    parser.add_argument('--dtype', choices = ['fp16', 'fp32', 'bf16'], default = 'fp32')
    parser.add_argument('--batch_size', default = 1, type = int)
    parser.add_argument('--max_length', default = 1000, type = int)

    return parser


def main():
    args = get_args().parse_args()
    accelerator = Accelerator()

    if args.save_path is None:
        spl_name = Path(args.data_path).stem.split('_')[-1]
        args.save_path = Path(args.ckpt_dir) / f'predictions_{spl_name}_{args.part}.jsonl'

    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if "pad_token" not in tokenizer.special_tokens_map:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
    
    model = load_hf_model_from_checkpoint(args.ckpt_dir, accelerator, args.dtype)
    
    # Dataset
    is_seq2seq = model.config.is_encoder_decoder
    dataset = load_test_dataset(args, tokenizer, is_seq2seq, args.part)

    # Generate function
    generator = SimpleGenerator(tokenizer, is_encoder_decoder = is_seq2seq)

    # Predict
    predictor = Predictor(
        accelerator, args.batch_size, 
        compute_preds_fn = generator, 
        collate_fn = SFT_Padding(tokenizer.pad_token_id, pad_side = 'left'),
    )
    all_preds = predictor.predict(model, dataset)
    
    # Post-process results
    if accelerator.is_main_process:
        
        if len(dataset) != len(all_preds):
            print((
                f'Warning: the number of generated samples ({len(all_preds)})' 
                f'is not equal to total data samples ({len(dataset)}). Truncate.'
            ))
            all_preds = all_preds[:len(dataset)]

        all_save = []
        for ori_d, pred_r in zip(dataset.data, all_preds):
            text = tokenizer.decode(pred_r['pred_tokens'], skip_special_tokens= True)
            save_d = {'title': ori_d['title'],
                      'para_idx': ori_d['para_idx'],
                      'q_id': ori_d['q_id'],
                      'type': ori_d['type'],
                      'prediction': text}
            all_save.append(save_d)

        # save
        save_jsonl(all_save, args.save_path)

if __name__ == '__main__':
    main()