import torch 
import re 
from misaki import en, espeak

from ..tiny_utils.display import bcolors

# Llama and Qwen
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

def colorFlush(token, color=bcolors.NICE):
    print(f"{color}{token}{bcolors.ENDC}",end="",flush=True)

# Kokoro 
def tokens_to_text(tokens):
    return ''.join(t.text + t.whitespace for t in tokens).strip()

def tokens_to_ps(tokens):
    return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()

def waterfall_last(tokens, next_count,
        waterfall = ['!.?…', ':;', ',—'], 
        bumps = [')', '”']):
    for w in waterfall:
        z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
        if z is None:
            continue
        z += 1
        if z < len(tokens) and tokens[z].phonemes in bumps:
            z += 1
        if next_count - len(tokens_to_ps(tokens[:z])) <= 510:
            return z
    return len(tokens)

def en_tokenize(tokens):
    tks = []
    pcount = 0
    for t in tokens:
        t.phonemes = '' if t.phonemes is None else t.phonemes
        next_ps = t.phonemes + (' ' if t.whitespace else '')
        next_pcount = pcount + len(next_ps.rstrip())
        if next_pcount > 510:
            z = waterfall_last(tks, next_pcount)
            text = tokens_to_text(tks[:z])
            ps = tokens_to_ps(tks[:z])
            yield text, ps, tks[:z]
            tks = tks[z:]
            pcount = len(tokens_to_ps(tks))
            if not tks:
                next_ps = next_ps.lstrip()
        tks.append(t)
        pcount += len(next_ps)
    if tks:
        text = tokens_to_text(tks)
        ps = tokens_to_ps(tks)
        yield ''.join(text).strip(), ''.join(ps).strip(), tks

def generate_audio(text: str, lang_code:str, pack: torch.Tensor, speed=1, model=None):
    fallback = espeak.EspeakFallback(british=False)
    g2p = en.G2P(trf=False, british=lang_code=='b', fallback=fallback, unk='')
    pattern = r'\n+'
    text = re.split(pattern, text.strip())
    for graphemes_index, graphemes in enumerate(text):
        if not graphemes.strip():
            continue
        if lang_code in 'ab':
            _, tokens = g2p(graphemes)
            for gs, ps, tks in en_tokenize(tokens):
                if not ps:
                    continue
                ps = ps[:510]
                audio, duration = model(ps, pack[len(ps)-1], speed)
                yield audio
            