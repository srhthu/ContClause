from transformers import PreTrainedModel
import accelerate
from accelerate import PartialState
from deepspeed.utils import safe_get_full_fp32_param
from pynvml import *

class DistLogger:
    """Logger for distributed environment"""
    def __init__(self):
        self.state = PartialState()
    
    def log_main(self, message):
        if self.state.local_process_index == 0:
            print('[Main]' + message)
    
    def log_process(self, message):
            print(f'[Process {self.state.local_process_index}] {message}')

logger = DistLogger()

def get_gpu_utilization():
    """Return GPU used RAM in MB"""
    nvmlInit()
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        yield i, info.used//1024**2

def print_gpu_utilization(logger: DistLogger = logger):
    for i, mem in get_gpu_utilization():
        logger.log_main(f'GPU {i} used {mem}MB')

def get_gpu_memory():
    all_gpu_mem = [k[1] for k in list(get_gpu_utilization())]
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES']
    if gpu_ids is None:
        state = accelerate.PartialState()
        gpu_ids = list(range(state.num_processes))
    else:
        gpu_ids = list(map(int, gpu_ids.split(',')))
    gpu_mem = [{k:all_gpu_mem[k]} for k in gpu_ids]
    return gpu_mem

def smart_resize_embeddings(tokenizer, model: PreTrainedModel, logger: DistLogger, verbose = False):
    """
    Resize the model input embeddings and synchronous the new token embeddings across devices
    """
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) <= old_vocab_size:
        # do not need resize embedding
        # for phi-1.5, the embedding size is 
        logger.log_main(f'Not resize embeddings. model: {old_vocab_size}, tokenizer: {len(tokenizer)}')
        return
    num_new_tokens = len(tokenizer) - old_vocab_size
    logger.log_main(f"Resize to add {num_new_tokens} new tokens")
    model.resize_token_embeddings(len(tokenizer))
    token_emb = model.get_input_embeddings()

    # broadcast new token embeddings
    new_embs_data = token_emb.weight.data[-num_new_tokens:]
    accelerate.utils.broadcast(new_embs_data, from_process = 0)
    if verbose:
        logger.log_process(f'last token emb: {token_emb.weight.data[-1,:10]}')

class ParamChangeChecker:
    """
    Check whether the parameter of model is updated (changed).

    Args:
        use_deepspeed: if use deepspeed, need get full param across devices.
    """
    def __init__(self, model, use_deepspeed = False):
        self.model = model
        self.use_deepspeed = use_deepspeed
        # select a parameter to check
        self.named_param = [(k,v) for k,v in model.named_parameters()][-1]

        self.last_observe = self.get_value()

        logger.log_main((
            f'check para: {self.named_param[0]},' 
            f'{self.last_observe.reshape(-1)[:3]}'
        ))
    
    def check(self):
        """Whether compare current param is different from previous observation"""
        cur_observe = self.get_value()
    
        diff = cur_observe - self.last_observe
        self.last_observe = cur_observe
        if diff.abs().sum() > 1e-6:
            return True
        else:
            return False
    
    def get_value(self):
        if self.use_deepspeed:
            cur_observe = safe_get_full_fp32_param(self.named_param[1]).clone().detach()
        else:
            cur_observe = self.named_param[1].clone().detach()

        return cur_observe