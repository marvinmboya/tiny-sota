import torch 
import torch.nn as nn 

from tiny_sota.models.kokoro_load import transferKokoroWeights, loadKokoroWeightsAndTok, check_generator_weights
from tiny_sota.models.kokoro_arch import KokoroConfig, Kokoro
from tiny_sota.tiny_utils.core import get_device
from misaki import en 
from tiny_sota.models.tiny_load import fetchFilesHuggingFace
from pathlib import Path

device = get_device()
config = KokoroConfig()
model = Kokoro(config)

import re
lang_code='a' 
g2p = en.G2P(trf=False, british=lang_code=='b', fallback=None, unk='')

def load_voice(voice: str):
    parent = Path.home()/".cache/tiny_sota"
    local_dir = parent/"models"
    fetchFilesHuggingFace(
        repo_id="hexgrad/Kokoro-82M",
        commit="8542409da2986c0ab5d41b3cf0411f7a58caab38",
        rem_id=f"voices/{voice}.pt",
        loc_id=f"{voice}.pt",
        local_dir=local_dir
    )
    voice_weight = local_dir/f"{voice}.pt"
    pack = torch.load(voice_weight, weights_only=True)
    return pack

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

def load_text(text: str, lang_code:str, pack: torch.Tensor, speed=1, model=None):
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

text = "The sky above the port was the color of television, tuned to a dead channel."
voice='af_heart'
pack = load_voice(voice).to(device)


weights = loadKokoroWeightsAndTok()
transferKokoroWeights(model, config, weights)
model.to(device)
model.eval()

import soundfile as sf 
text = "In the end, everything sort of becomes a web of strings and a pot of chaos."
for i, audio in enumerate(load_text(text, lang_code, pack, model=model)):
    sf.write(f'{i}.wav', audio, 24000)


