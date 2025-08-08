from dataclasses import dataclass
import torch 

@dataclass
class GenerateConfig:
    context_len: int
    device: torch.device 
    max_new_tokens: int = 500 

def generate_text_stream(
        model, token_ids, max_new_tokens, context_len, 
        temperature=0.0, top_k=None, eos_token_id=None, is_hf_model=False):
    for _ in range(max_new_tokens):
        idx_cond = token_ids[:, -context_len:]
        with torch.no_grad():
            if is_hf_model:
                logits = model(idx_cond).logits
            else:
                logits = model(idx_cond)
            logits = logits[:, -1]
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