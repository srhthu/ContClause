import torch
import torch.nn as nn

def compute_clm_loss_with_ignore(model, batch, ignore_index):
    # out = model(**batch)
    out = model(input_ids = batch['input_ids'])
    logits = out['logits']
    labels = batch['input_ids']
    
    # Copy from Llama
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    return loss


class LM_Loss_With_Ignore:
    def __init__(self, ignore_index):
        self.ignore_index = ignore_index
    
    def __call__(self, model, batch):
        loss = compute_clm_loss_with_ignore(model, batch, self.ignore_index)
        return {'loss': loss}

class LM_Simple_Feed:
    def __call__(self, model, batch):
        batch = {k:v for k,v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        loss = model(**batch).loss
        return {'loss': loss}