import torch
from .tiny_load import (
    MODELS_META, 
    getLocalDir, 
    fetchFilesHuggingFace
)
from ..meta import KOKORO_VOICES

def loadKokoroWeightsAndVoice(voice: str = KOKORO_VOICES.FEMALE.HEART):
    meta = MODELS_META.Kokoro_82M
    local_dir = getLocalDir()
    loc_weights = fetchFilesHuggingFace(
        repo_id = meta["repo_id"],
        commit = meta["commit"],
        rem_id = meta["weight_id"],
        loc_id = meta["loc_weight"],
        local_dir=local_dir
    )
    local_dir = getLocalDir(dir="voices")
    loc_voice = fetchFilesHuggingFace(
        repo_id = meta["repo_id"],
        commit = None,
        rem_id = f"voices/{voice}.pt",
        loc_id = f"{voice}.pt",
        local_dir=local_dir
    )
    weights_dict = torch.load(
        loc_weights, map_location="cpu", weights_only=True)
    voice = torch.load(loc_voice, weights_only=True)
    return weights_dict, voice