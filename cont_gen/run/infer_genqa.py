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
    GenerationMixin, PreTrainedModel
)
from accelerate import Accelerator

from cont_gen.model_utils import load_hf_model_from_checkpoint
from cont_gen.data_loader.cuad_prompt import CUAD_SFT, CUAD_SFT_Seq2Seq, SFT_Padding
from cont_gen.trainer.train_only_accelerate import Predictor

MAX_NEW_TOKENS = 512

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
        

class GeneratorWithProbNone:
    """Generate on prompts and """
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        neg_token_ids = tokenizer(['None'], return_tensors = 'pt')
        self.neg_token_ids = neg_token_ids.input_ids.to(device)
        self.gen_config = GenerationConfig(
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample = False,

        )
    def __call__(self, model: PreTrainedModel, batch):
        outputs = model.generate(batch['input_ids'], generation_config = self.gen_config)
    
    
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--prompt_path')
    parser.add_argument('--save_path')
    parser.add_argument('--base_model')
    parser.add_argument('--ckpt_dir', help = 'directory of model checkpoint')
    parser.add_argument('--dtype', choices = ['fp16', 'fp32', 'bf16'], default = 'fp32')
    parser.add_argument('--batch_size', default = 1, type = int)
    parser.add_argument('--max_length', default = 1200, type = int)

    return parser


def main():
    args = get_args().parse_args()
    accelerator = Accelerator()

    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = load_hf_model_from_checkpoint(args.ckpt_dir, accelerator, args.dtype)
    
    # Dataset
    ds_cls = CUAD_SFT_Seq2Seq if model.config.is_encoder_decoder else CUAD_SFT
    dataset = ds_cls(args.prompt_path, tokenizer, args.max_length, 
                     is_test = True, small = args.debug)

    # Generate function
    generator = SimpleGenerator(tokenizer, is_encoder_decoder = model.config.is_encoder_decoder)

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
                f'is not equal to total data samples ({len(dataset)})'
            ))

        all_save = []
        for ori_d, pred_r in zip(dataset.data, all_preds):
            text = tokenizer.decode(pred_r['pred_tokens'], skip_special_tokens= True)
            save_d = {
                'title': ori_d['title'],
                'chunk_index': ori_d['chunk_index'],
                'q_id': ori_d['q_id'],
                'prediction': text
            }
            all_save.append(save_d)

        # save
        with open(args.save_path, 'w') as f:
            for d in all_save:
                f.write(json.dumps(d, ensure_ascii = False) + '\n')

if __name__ == '__main__':
    main()