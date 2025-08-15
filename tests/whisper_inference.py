import torch 
from pathlib import Path 

from tiny_sota.models import (
    Whisper, transferWhisperWeights, Configs
)

from tiny_sota.models.whisper_load import (
    load_audio, log_mel_spectrogram,
    N_SAMPLES, N_FRAMES, HOP_LENGTH, 
    SAMPLE_RATE, FRAMES_PER_SECOND, transcribe 
)

from tiny_sota.models.whisper_tok import get_tokenizer
from tiny_sota.models.whisper_meta import SpeechOptions, DecodeOptions
from tiny_sota.tiny_utils import get_device

device = get_device()

config = Configs.Whisper()
model = Whisper(config)

path = Path.home()/"Downloads/small.pt"
pretrained_model = torch.load(path, weights_only=False)
pretrained_weights = pretrained_model['model_state_dict']
transferWhisperWeights(model, config, pretrained_weights)
del pretrained_model, pretrained_weights
model.to(device)

def run_whisper(audio_path, *, language="en", task="transcribe"):
    tokenizer = get_tokenizer(language=language, task=task)
    audio = load_audio(audio_path)
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

run_whisper("./files/english.wav")
run_whisper("./files/japanese.mp3")
run_whisper("./files/japanese.mp3", task="translate")


