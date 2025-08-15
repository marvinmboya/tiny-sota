import torch 
import sys 
from pathlib import Path 

from tiny_sota.models import (
    Whisper, transferWhisperWeights, ModelConfigs
)
from tiny_sota.models.whisper_load import fetchWhisperSmallWeights
from tiny_sota.models import loadWhisperSmallWeightsAndTok

from tiny_sota.tiny_utils import get_device
from tiny_sota.models.configs import Audio_Transcribe_Params

device = get_device()

config = ModelConfigs.Whisper
model = Whisper(config)

fetchWhisperSmallWeights()
loadWhisperSmallWeightsAndTok(Audio_Transcribe_Params)
