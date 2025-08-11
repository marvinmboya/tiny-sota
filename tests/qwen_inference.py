import torch
import sys
import torch.nn as nn


from tiny_sota.models import (
    transferQwen3Weights,
    loadQwen3WeightsAndTok,
    Qwen3Model, Configs
)
from tiny_sota.models.tiny_load import getLocalWeightsDir
from tiny_sota.inference import TokenizerChoices, LLMEngine
from tiny_sota.tiny_utils import get_device

device = get_device()
parent = getLocalWeightsDir()
config = Configs.Qwen

model = Qwen3Model(config)
weights, tok = loadQwen3WeightsAndTok()
transferQwen3Weights(model, config, weights)
del weights

model.to(device)

prompt = "write a nice little matrix multiplication example in CUDA"
engine = LLMEngine(model, tok,
        TokenizerChoices.qwen,
        add_generation_prompt = True,
        think_mode = False)
engine(prompt)

"""
LOAD MODEL WITHOUT WEIGHTS FROM HUGGINGFACE
s_config =  AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
model = Qwen3ForCausalLM._from_config(s_config)
"""
