import torch 
import re, subprocess, sys 
from misaki import en, espeak

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

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

def generate_audio(model, g2p, text, voice, lang_code, speed=1):
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
                audio, duration = model(ps, voice[len(ps)-1], speed)
                yield audio
        else:
            chunk_size = 400
            chunks = []            
            sentences = re.split(r'([.!?]+)', graphemes)
            current_chunk = ""
            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]                    
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk.strip())
            if not chunks:
                chunks = [graphemes[i:i+chunk_size] for i in range(0, len(graphemes), chunk_size)]
            for chunk in chunks:
                if not chunk.strip():
                    continue
                ps, _ = g2p(chunk)
                if not ps:
                    continue
                ps = ps[:510]
                audio, duration = model(ps, voice[len(ps)-1], speed)
                yield audio          


def set_g2p(lang_code: str):
    _codes = {'e':'es','f':'fr-fr','h':'hi','i':'it','p':'pt-br'}
    g2p = None
    if lang_code in 'ab':
        fallback = espeak.EspeakFallback(british=lang_code=='b')
        g2p = en.G2P(trf=False, british=lang_code=='b', fallback=fallback, unk='')
    elif lang_code == 'j':
        install("misaki[ja]")
        from misaki import ja
        g2p = ja.JAG2P()
    elif lang_code == 'z':
        install("misaki[zh]")
        from misaki import zh
        g2p = zh.ZHG2P(version=None, en_callable=None)
    else:
        language = _codes[lang_code]
        g2p = espeak.EspeakG2P(language=language)
        lang_code = language
    return g2p, lang_code
