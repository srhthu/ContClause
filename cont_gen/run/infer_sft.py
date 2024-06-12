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

from cont_gen.utils.model_utils import load_hf_model_from_checkpoint, get_model_path_from_ckpt
from cont_gen.data_loader.cuad_prompt import SFT_Padding
from cont_gen.data_loader.cuad_sft import CUAD_SFT_Cached, CUAD_SFT_Filter_Type
from cont_gen.trainer.train_only_accelerate import Predictor
from cont_gen.utils import save_jsonl, get_ckpt_paths

MAX_NEW_TOKENS = 600

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
        is_chat = args.is_chat,
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

        if 'Meta-Llama-3-8B-Instruct' in tokenizer.name_or_path:
            eos_token_id = [128001, 128009]
        else:
            eos_token_id = tokenizer.eos_token_id

        self.gen_config = GenerationConfig(
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample = False,
            eos_token_id = eos_token_id,
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
    parser.add_argument('--is_chat', action = 'store_true')
    parser.add_argument('--is_seq2seq', action = 'store_true')
    parser.add_argument('--base_model', help = 'specify base model to load tokenizer')
    parser.add_argument('--ckpt_dir', help = 'directory of model checkpoint or peft checkpoint')
    parser.add_argument('--run_dir', help = 'infer of all checkpoints udner run_dir')
    parser.add_argument('--save_prefix', default = '', 
                        help = "prefix of saved file name. Targeted for model trained on all labels")
    parser.add_argument('--dtype', choices = ['fp16', 'fp32', 'bf16'], default = 'fp32')
    parser.add_argument('--batch_size', default = 1, type = int)
    parser.add_argument('--max_length', default = 1000, type = int)

    return parser


def handle_one_ckpt(ckpt, dataset, model, predictor, accelerator, tokenizer, save_path = None, save_name = None):
    """
    Do inference and save results.
    """
    if save_path is None:
        assert save_name is not None
        save_path = Path(ckpt) / save_name

    if Path(save_path).exists():
        print('predictions exist.')
        return
    
    # Predict
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
        save_jsonl(all_save, save_path)
    

def main():
    args = get_args().parse_args()
    accelerator = Accelerator()

    # determin ckpt dirs
    if args.ckpt_dir is not None:
        ckpts = [args.ckpt_dir]
    else:
        assert args.run_dir is not None
        ckpts = [str(k) for k in get_ckpt_paths(args.run_dir)]

    save_name  = None
    if args.save_path is None:
        spl_name = Path(args.data_path).stem.split('_')[-1]
        save_name = f'{args.save_prefix}predictions_{spl_name}_{args.part}.jsonl'

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model if args.base_model else get_model_path_from_ckpt(ckpts[0]))
    if "pad_token" not in tokenizer.special_tokens_map:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset
    dataset = load_test_dataset(args, tokenizer, args.is_seq2seq, args.part)

    # Generate function
    generator = SimpleGenerator(tokenizer, is_encoder_decoder = args.is_seq2seq)

    # Predictor
    predictor = Predictor(
        accelerator, args.batch_size, 
        compute_preds_fn = generator, 
        collate_fn = SFT_Padding(tokenizer.pad_token_id, pad_side = 'left'),
    )

    

    for ckpt in ckpts:
        print(f'Handle checkpoint: {str(ckpt)}')
        model = load_hf_model_from_checkpoint(ckpt, accelerator, args.dtype)

        handle_one_ckpt(ckpt, dataset, model, predictor, accelerator, tokenizer, 
                        save_name = save_name,
                        save_path = args.save_path)
        
        del model

    

if __name__ == '__main__':
    main()