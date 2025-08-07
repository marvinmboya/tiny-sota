import torch, torch.nn as nn 
from safetensors.torch import load_file
from .tiny_load import (
    LLMS_META, getLocalWeightsDir, fetchLLMWeightAndTok
)

def fetchLlamaWeightsAndTok():
    Llama_Meta = LLMS_META.Llama3_2_1B
    local_dir = getLocalWeightsDir()
    fetchLLMWeightAndTok(Llama_Meta, local_dir)

def loadLlamaWeightsAndTok():
    Llama_Meta = LLMS_META.Llama3_2_1B
    local_dir = getLocalWeightsDir()
    loc_weight  = local_dir/Llama_Meta["loc_weight"]
    loc_tok  = local_dir/Llama_Meta["loc_tok"]
    assert loc_weight.exists(), "Llama weights not downloaded!"
    assert loc_tok.exists(), "Llama tokenizer not downloaded!"
    weight_dict = load_file(loc_weight)
    return weight_dict, loc_tok