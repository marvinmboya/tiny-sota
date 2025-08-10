import torch
import sys
import torch.nn as nn


from tiny_sota.models import (
    transferQwenWeights,
    loadQwenWeightsAndTok,
    Qwen3Model, Configs
)
from tiny_sota.models.tiny_load import getLocalWeightsDir
from tiny_sota.inference import GenerateConfig, TokenizerChoices, LLMEngine

from tiny_sota.tiny_utils import get_device

device = get_device()
parent = getLocalWeightsDir()
config = Configs.Qwen

model = Qwen3Model(config)
weights, tok = loadQwenWeightsAndTok()
transferQwenWeights(model, config, weights)
del weights

model.to(device)

generate_config = GenerateConfig(
     context_len=config.context_len,
     max_new_tokens=2000,
     device = device)

prompt = "write a nice little matrix multiplication example in CUDA"
engine = LLMEngine(model, tok,
                   TokenizerChoices.qwen,
                   add_generation_prompt = True,
                   think_mode = False)
engine(prompt, generate_config,is_hf_model=False)

"""
LOAD MODEL WITHOUT WEIGHTS FROM HUGGINGFACE
s_config =  AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
model = Qwen3ForCausalLM._from_config(s_config)
"""
