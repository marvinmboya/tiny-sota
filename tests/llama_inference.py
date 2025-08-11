import torch
import sys
import torch.nn as nn


from tiny_sota.models import (
    fetchLlama3WeightsAndTok,
    loadLlama3WeightsAndTok,
    transferLlama3Weights,
    Llama3Model, Configs
)

from tiny_sota.models.tiny_load import getLocalWeightsDir
from tiny_sota.inference import TokenizerChoices, LLMEngine
from tiny_sota.tiny_utils import get_device

device = get_device()
parent = getLocalWeightsDir()
config = Configs.Llama

model = Llama3Model(config)

fetchLlama3WeightsAndTok()
weights, tok = loadLlama3WeightsAndTok()
transferLlama3Weights(model, config, weights)
del weights
model.to(device)

prompt = "the story of the princess and the lost castle"
engine = LLMEngine(
          model, tok, TokenizerChoices.qwen,
          add_generation_prompt = True,
          think_mode = False)
engine(prompt)
