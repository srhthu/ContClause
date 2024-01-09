"""Utilities for build, load and prepare model"""
from typing import List, Tuple, Union

from transformers import (
    AutoTokenizer, AutoConfig, PreTrainedModel,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM
)
import accelerate
from accelerate import Accelerator
from peft import  (
    LoraConfig, get_peft_model, PeftModel
)

from cont_gen.trainer.utils_dist import DistLogger

def build_hf_or_peft_model(
    base_model, accelerator: Accelerator, torch_dtype, peft_config = None
):
    # prepare arguments for from_pretrained
    config = AutoConfig.from_pretrained(base_model, trust_remote_code = True)
    kws = dict(
        torch_dtype = torch_dtype,
        device_map = accelerator.local_process_index,
        trust_remote_code = True
    )
    is_zero3 = False if accelerator.state.deepspeed_plugin is None else \
            accelerator.state.deepspeed_plugin.zero3_init_flag
    if is_zero3:
        _ = kws.pop('device_map')
    # add model specific parameters
    if 'phi' in base_model:
        # use flash attention for microsoft/phi-1_5
        kws.update(dict(flash_attn=True, flash_rotary=True, fused_dense=True))
    
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model, **kws)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, **kws)
    
    if peft_config is not None:
        task_type = "SEQ_2_SEQ_LM" if config.is_encoder_decoder else "CAUSAL_LM"
        peft_config.task_type = task_type
        model = get_peft_model(model, config)
    
    return model

def smart_resize_embeddings(
    tokenizer, 
    model: Union[PreTrainedModel, PeftModel], 
    logger: DistLogger, 
    verbose = False
):
    """
    Resize the model input embeddings and synchronous the new token embeddings across devices
    """
    base_model = model if isinstance(model, PreTrainedModel) \
                else model.base_model if isinstance(model, PeftModel) \
                else None
    old_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) <= old_vocab_size:
        # do not need resize embedding
        logger.log(f'Not resize embeddings. model: {old_vocab_size}, tokenizer: {len(tokenizer)}')
        return
    num_new_tokens = len(tokenizer) - old_vocab_size
    logger.log(f"Resize to add {num_new_tokens} new tokens")
    base_model.resize_token_embeddings(len(tokenizer))
    token_emb = base_model.get_input_embeddings()

    # broadcast new token embeddings
    new_embs_data = token_emb.weight.data[-num_new_tokens:]
    accelerate.utils.broadcast(new_embs_data, from_process = 0)
    if verbose:
        logger.log_process(f'last token emb: {token_emb.weight.data[-1,:10]}')