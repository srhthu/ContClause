"""Utilities for build, load and prepare model"""
from typing import List, Tuple, Union
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoConfig, PreTrainedModel,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedType
from peft import  (
    LoraConfig, get_peft_model, PeftModel, load_peft_weights
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from cont_gen.utils import load_json
from cont_gen.trainer.utils_dist import DistLogger

TORCH_DTYPE_MAP = {
    'fp16': torch.float16, 
    'bf16': torch.bfloat16, 
    'fp32': torch.float32
}

def build_hf_or_peft_model(
    base_model, 
    accelerator: Accelerator, 
    torch_dtype: Union[torch.dtype, str], 
    quantization = False,
    peft_config = None
):
    # prepare arguments for from_pretrained
    config = AutoConfig.from_pretrained(base_model, trust_remote_code = True)
    torch_dtype = torch_dtype if isinstance(torch_dtype, torch.dtype) \
                    else TORCH_DTYPE_MAP[torch_dtype]
    
    # Build model. Determin arguments for from_pretrained
    ## Handle device_map based on distributed_type
    if accelerator.state.distributed_type == DistributedType.NO:
        # model parallelism
        device_map = 'auto'
    else:
        device_map = accelerator.local_process_index

    kws = dict(
        torch_dtype = torch_dtype,
        device_map = device_map,
        trust_remote_code = True
    )
    ## Do not specify device_map for zero3 
    is_zero3 = False if accelerator.state.deepspeed_plugin is None else \
            accelerator.state.deepspeed_plugin.zero3_init_flag
    if is_zero3:
        _ = kws.pop('device_map')
    ## Add model specific parameters
    if 'phi' in base_model:
        # use flash attention for microsoft/phi-1_5
        kws.update(dict(flash_attn=True, flash_rotary=True, fused_dense=True))
    
    ## Add quantization config
    if quantization:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kws['quantization_config'] = quant_config
    
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model, **kws)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, **kws)
    
    if peft_config is not None:
        task_type = "SEQ_2_SEQ_LM" if config.is_encoder_decoder else "CAUSAL_LM"
        peft_config.task_type = task_type
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model

def load_hf_model_from_checkpoint(ckpt_dir, accelerator: Accelerator, torch_dtype: Union[str, torch.dtype], base_model = None):
    """
    Load huggingface model from checkpoint saved by save_pretrained. Support Lora.

    To resolve the embedding resize issue, we first load base model, then resize, then load peft.
    """
    # try to find peft config
    is_peft = False
    if (Path(ckpt_dir) / 'adapter_config.json').exists():
        is_peft = True
        adap_cfg_dt = load_json(Path(ckpt_dir) / 'adapter_config.json')
        if base_model is None:
            base_model = adap_cfg_dt['base_model_name_or_path']
    else:
        base_model = ckpt_dir
    
    # load base model
    torch_dtype = torch_dtype if isinstance(torch_dtype, torch.dtype) \
                    else TORCH_DTYPE_MAP[torch_dtype]
    config = AutoConfig.from_pretrained(base_model, trust_remote_code = True)
    kws = dict(
        torch_dtype = torch_dtype,
        device_map = accelerator.local_process_index,
        trust_remote_code = True
    )
    if 'phi' in ckpt_dir:
        # use flash attention for microsoft/phi-1_5
        kws.update(dict(flash_attn=True, flash_rotary=True, fused_dense=True))
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model, **kws)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, **kws)
    
    if not is_peft:
        return model
    
    # determin whether to resize embedding.
    ipt_emb = model.get_input_embeddings()
    _, ipt_emb_w = list(ipt_emb.named_parameters())[0]
    ## find input_emb name
    emb_w_name = [n for n,p in model.named_parameters() if p is ipt_emb_w][0]

    ## Get saved emb
    adapter_st = load_peft_weights(ckpt_dir)
    prefix = "base_model.model."
    saved_emb = adapter_st.get(prefix + emb_w_name)
    if saved_emb is None:
        print('embedding is not saved by peft')
        model.load_adapter(ckpt_dir)
        return model

    ## Resize emb
    old_size = list(ipt_emb.parameters())[0].shape[0]
    new_size = saved_emb.shape[0]
    if old_size == new_size:
        print('Do not need resize emb')
        model.load_adapter(ckpt_dir)
        return model
    
    print(f'Resize embedding num to {new_size}')
    model.resize_token_embeddings(new_size)
    model.load_adapter(ckpt_dir)

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