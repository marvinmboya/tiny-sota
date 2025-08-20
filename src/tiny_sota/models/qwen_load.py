import torch, torch.nn as nn 
from safetensors.torch import load_file
from .tiny_load import (
    assign, MODELS_META, getLocalWeightsDir, fetchFilesHuggingFace
)
from ..tokenizers.qwen import Qwen3Tokenizer
from .configs import Qwen_Tok_Options

def loadQwen3WeightsAndTok(qwen_tok_options: Qwen_Tok_Options):
    meta = MODELS_META.Qwen3_06B
    local_dir = getLocalWeightsDir()
    loc_weight = fetchFilesHuggingFace(
        repo_id = meta["repo_id"],
        commit = meta["commit"],
        rem_id = meta["weight_id"],
        loc_id = meta["loc_weight"],
        local_dir=local_dir
    )
    local_dir = getLocalWeightsDir(dir="tokenizers")
    loc_tok = fetchFilesHuggingFace(
        repo_id = meta["repo_id"],
        commit = meta["commit"],
        rem_id = meta["tok_id"],
        loc_id = meta["loc_tok"],
        local_dir=local_dir
    )
    weight_dict = load_file(loc_weight)
    tokenizer = Qwen3Tokenizer(
        loc_tok, 
        add_generation_prompt = qwen_tok_options.add_generation_prompt, 
        think_mode = qwen_tok_options.think_mode
    )
    return weight_dict, tokenizer