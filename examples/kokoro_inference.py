import torch 
import torch.nn as nn 

from tiny_sota.models.kokoro_load import loadKokoroWeightsAndVoice
from tiny_sota.transfers import transferKokoroWeights
from tiny_sota.models.kokoro_arch import KokoroConfig, Kokoro
from tiny_sota.tiny_utils.core import get_device
from tiny_sota.inference.utils import generate_audio
from tiny_sota.meta import KOKORO_VOICES

import soundfile as sf 

device = "cpu"
config = KokoroConfig()
model = Kokoro(config, device=device)

weights, voice = loadKokoroWeightsAndVoice(KOKORO_VOICES.MALE.FENRIR)
transferKokoroWeights(model, config, weights)
model.to(device)
model.eval()
pack = voice.to(device)
del weights 

lang_code='a'
text = "Hi, hello. So we know that this channel is about self-expression, correct? We know that this channel is not giving therapy, it's not giving 100% correct advice, it is things that are rooted in my own life experience and things that I have learned from life, and it's advice that you can either take or leave and you need to listen to everything I say with a grain of salt. We're aware of this, that the internet is not a place you go for absolute truths, it's a place you go for perspectives, it's a place you go for opening your mind, looking at things a little bit differently. We're aware of that, right? I want to talk about the truth of being yourself."
for i, audio in enumerate(generate_audio(text, lang_code, pack, model=model)):
    sf.write(f'{i}.wav', audio, 24000)


