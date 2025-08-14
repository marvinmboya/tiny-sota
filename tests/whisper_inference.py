import torch 
from pathlib import Path 

from tiny_sota.models import (
    Whisper, transferWhisperWeights, Configs
)

from tiny_sota.models.whisper_load import (
    load_audio, log_mel_spectrogram, get_tokenizer,
    N_SAMPLES, N_FRAMES, HOP_LENGTH, 
    SAMPLE_RATE, FRAMES_PER_SECOND, transcribe 
)
from tiny_sota.models.whisper_meta import SpeechOptions, DecodeOptions
from tiny_sota.tiny_utils import get_device

device = get_device()

@torch.no_grad()
def infer(model, x):
    model.eval()
    out = model(x)
    return out

config = Configs.Whisper()
model = Whisper(config)

path = Path.home()/"Downloads/small.pt"
pretrained_model = torch.load(path, weights_only=False)
pretrained_weights = pretrained_model['model_state_dict']
transferWhisperWeights(model, config, pretrained_weights)
del pretrained_model, pretrained_weights
model.to(device)

tokenizer = get_tokenizer(language="en")
audio = load_audio("./english.wav")
mel = log_mel_spectrogram(audio, config.n_mels, padding=N_SAMPLES)

out = transcribe(
    model = model, 
    tokenizer=tokenizer, 
    mel = mel, 
    config = config, 
    speech_options=SpeechOptions(),
    decode_options=DecodeOptions()
)

print(out['text'])
import sys; sys.exit(0)
audio = load_audio("english.wav")
content_frames = mel.shape[-1] - N_FRAMES
content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)


