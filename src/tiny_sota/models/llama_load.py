from safetensors.torch import load_file
from .tiny_load import (
    MODELS_META, getLocalWeightsDir, fetchFilesHuggingFace
)
from ..tokenizers.llama import Llama3Tokenizer

def loadLlama3WeightsAndTok():
    meta = MODELS_META.Llama32_1B
    local_dir = getLocalWeightsDir()
    loc_weight = fetchFilesHuggingFace(
        repo_id = meta["repo_id"],
        commit = meta["commit"],
        rem_id = meta["weight_id"],
        loc_id = meta["loc_weight"],
        local_dir=local_dir
    )
    loc_tok = fetchFilesHuggingFace(
        repo_id = meta["repo_id"],
        commit = meta["commit"],
        rem_id = meta["tok_id"],
        loc_id = meta["loc_tok"],
        local_dir=local_dir
    )
    weight_dict = load_file(loc_weight)
    tokenizer = Llama3Tokenizer(loc_tok)
    return weight_dict, tokenizer
