"""
For memory enhanced generative qa, we process the paragraphs one by one and ask all questions. Then extract clause types for memory.
"""
from typing import List
import json
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, GenerationConfig, AutoTokenizer
from accelerate import Accelerator

from cont_gen.data_loader.cuad_prompt import SFT_Padding
from cont_gen.data_process.utils import tokenize_wo_eos
from cont_gen.data_process.build_genqa_with_context import (
    get_head, build_inputs_with_both
)
from cont_gen.utils import (
    load_json, load_jsonl, save_jsonl, save_jsonl_append
)
from cont_gen.model_utils import load_hf_model_from_checkpoint

MAX_NEW_TOKENS = 512

def infer_batch_data(sources: List[str], model, tokenizer, 
                     max_length, batch_size)-> List[torch.Tensor]:
    """
    Given some text, generate using the model.

    This can be configured to run on one device, or run with distributed devices.
    We implement the simple version of running on one device
    """
    # prepare input features
    inputs = []
    for source in sources:
        # We remove the eos token of the source sequence, which is not 
        # necessary for t5. To keep the consistency, we also remove the eos token
        # during inference. In future version, this may be fixed.
        # enc = tokenizer(source, truncation = True, max_length = max_length)
        enc = tokenize_wo_eos(tokenizer, source, 
                              truncation = True, max_length = max_length)
        inputs.append({'input_ids': enc.input_ids, 
                       'attention_mask': enc.attention_mask})
    # build dataloader for batch inference
    dl = DataLoader(inputs, batch_size = batch_size, 
                    shuffle=False, drop_last=False, 
                    collate_fn= SFT_Padding(tokenizer.pad_token_id))
    # build generation config
    gen_config = GenerationConfig(
        max_new_tokens = MAX_NEW_TOKENS,
        do_sample = False,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id
    )
    results = []
    model.eval()
    for batch in dl:
        batch = {k: v.cuda() for k,v in batch.items()}
        input_length = 1 if model.config.is_encoder_decoder \
                        else batch['input_ids'].shape[1]
        # infer
        with torch.no_grad():
            outputs = model.generate(**batch, generation_config = gen_config)
        generated_tokens = outputs[:, input_length:].cpu()
        results.extend(generated_tokens)
    return results

def infer_one_doc(doc_data, model, tokenizer: PreTrainedTokenizer, 
                  clause_types, quest_list, 
                  max_mem_num, total_mem_len, max_length, batch_size,
                  tqdm_bar = None):
    para_preds = []
    all_mem_heads = []
    all_mem_clauses = []
    for para_idx, para in enumerate(doc_data['paras']):
        p_text = para['text']

        # add head to memory
        all_mem_heads.append(get_head(p_text))

        # prepare context memory
        mem_first = max(para_idx - max_mem_num, 0)
        ctx_mem_heads = all_mem_heads[mem_first: para_idx]
        ctx_mem_clauses = all_mem_clauses[mem_first: para_idx]

        # ask all questions
        # build input text
        sources = []
        for quest in quest_list:
            source, _ = build_inputs_with_both(
                tokenizer, p_text, quest, answers = None,
                memory_head = ctx_mem_heads,
                memory_clauses = ctx_mem_clauses,
                total_memory_len = total_mem_len,
                max_answer_len= 60 # this is not used
            )
            sources.append(source)
        # infer with model
        pred_tokens = infer_batch_data(sources, model, tokenizer, 
                                       max_length, batch_size)
        # parse token_ids to text
        pred_text = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        # prepare prediction data
        for qi, pt in enumerate(pred_text):
            para_preds.append({'title': doc_data['title'],
                               'para_index': para_idx,
                               'q_id': qi,
                               'prediction': pt,
                               #'source': sources[qi]
                               })
            # add the source field for debug
        # prepare memory of clauses
        mem_cla = [clause_types[qi] for qi, pt in enumerate(pred_text) \
                    if not pt.startswith('None')]
        all_mem_clauses.append(mem_cla)
        if tqdm_bar is not None:
            tqdm_bar.update()
    
    return para_preds

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--test_path', help = 'path of test paragraph data')
    parser.add_argument('--save_path', help = 'path to save predictions')
    parser.add_argument('--model_ckpt', help = 'path of trained model')
    parser.add_argument('--base_model', help = 'path of base model for tokenizer')
    parser.add_argument('--dtype', default = 'fp32')
    parser.add_argument('--clause_names', 
                        default = 'data/clause/ori_clause_names.json',
                        help = 'path of clause names')
    parser.add_argument('--quests', 
                        default = 'data/clause/prompt_quest.json',
                        help = 'path of clause questions')
    # max_mem_num, total_mem_len, max_length, batch_size
    parser.add_argument('--max_mem_num', default = 10, type = int, 
                        help = 'number of max pieces of memory')
    parser.add_argument('--total_mem_len', default = 256, type = int, 
                        help = 'max token length of memory context')
    parser.add_argument('--max_length', default = 1000, type = int, 
                        help = 'max token length of source text')
    parser.add_argument('--batch_size', default = 6, type = int,
                        help = 'inference device batch size')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(json.dumps(args.__dict__, ensure_ascii=False, indent = 4))

    # load data
    clause_names = load_json(args.clause_names)
    quests = load_json(args.quests)

    test_data = load_jsonl(args.test_path)

    # build model and tokenizer
    model = load_hf_model_from_checkpoint(args.model_ckpt, Accelerator(),args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, 
                                              trust_remote_code = True)
    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    part_test_data = test_data
    total_para = sum([len(d['paras']) for d in part_test_data])
    bar = tqdm(total = total_para, ncols = 80)
    for cont_data in part_test_data:
        predictions = infer_one_doc(cont_data, model, tokenizer,
                      clause_types = clause_names,
                      quest_list = quests,
                      max_mem_num = args.max_mem_num,
                      total_mem_len = args.total_mem_len,
                      max_length = args.max_length,
                      batch_size = args.batch_size,
                      tqdm_bar = bar)
        save_jsonl_append(predictions, args.save_path)
    bar.close()
    

if __name__ == '__main__':
    main()