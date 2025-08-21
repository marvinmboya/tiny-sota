import torch
import sys
import torch.nn as nn


from tiny_sota.models import (
    loadLlama3WeightsAndTok,
    Llama3Model, ModelConfigs
)
from tiny_sota.transfers import transferLlama3Weights

from tiny_sota.models.tiny_load import getLocalDir
from tiny_sota.inference import LLMEngine
from tiny_sota.tiny_utils import get_device

device = get_device()
parent = getLocalDir()
config = ModelConfigs.Llama

model = Llama3Model(config)

weights, tok = loadLlama3WeightsAndTok()
transferLlama3Weights(model, config, weights)
del weights

prompt = "the story of the princess and the lost castle"
engine = LLMEngine(model, tok, device)
engine(prompt)
