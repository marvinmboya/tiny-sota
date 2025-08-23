import torch
from .tiny_load import (
    MODELS_META, 
    getLocalDir, 
    fetchFilesHuggingFace
)
from dataclasses import dataclass
from ..meta import KOKORO_VOICES

@dataclass
class VoicePack:
    person: str 
    lang_code: str
    voice: torch.Tensor 

def loadKokoroWeightsAndVoice(voice: str = KOKORO_VOICES.FEMALE.AMERICAN_ENGLISH):
    meta = MODELS_META.Kokoro_82M
    lang_code = voice[0]
    person = voice.split("_")[-1]
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
    print(type(weights_dict))
    voice_pack = VoicePack(
        person=person, 
        lang_code=lang_code, 
        voice=voice
    )
    return weights_dict, voice_pack