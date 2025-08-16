import torch.nn as nn

from tiny_sota.models import (
    transferQwen3Weights,
    loadQwen3WeightsAndTok,
    Qwen3Model, ModelConfigs
)

from tiny_sota.inference import LLMEngine
from tiny_sota.tiny_utils import get_device
from tiny_sota.models.configs import Qwen_Tok_Options

device = get_device()
config = ModelConfigs.Qwen
model = Qwen3Model(config)

weights, tok = loadQwen3WeightsAndTok(
    Qwen_Tok_Options(
        add_generation_prompt=True,
        think_mode=False
    )
)
transferQwen3Weights(model, config, weights)
del weights

prompt = "the story of the princess and the lost castle"
engine = LLMEngine(model, tok, device)
engine(prompt)

