from dataclasses import dataclass
import torch 

from ..tiny_utils import get_device 

def generate_text_stream(
        model, token_ids, max_new_tokens, 
        temperature=0.0, top_k=None, eos_token_id=None):
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(token_ids)[:, -1]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        if (eos_token_id is not None
                   and torch.all(next_token_id == eos_token_id)):
            break
        yield next_token_id
        token_ids = torch.cat((token_ids, next_token_id), dim=1)
