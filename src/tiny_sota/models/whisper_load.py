import torch
from .tiny_load import (
    MODELS_META, getLocalWeightsDir, fetchGenericFiles
)
from ..tokenizers.whisper import get_tokenizer
from .configs import Audio_Transcribe_Params

def fetchWhisperSmallWeights():
    Whisper_Meta = MODELS_META.Whisper_Small
    local_dir = getLocalWeightsDir()
    loc_weights = fetchGenericFiles(
        Whisper_Meta["url"], 
        local_dir, 
        Whisper_Meta["loc_weight"])
    return loc_weights

def loadWhisperSmallWeightsAndTok(audio_transcribe_params: Audio_Transcribe_Params):
    loc_weight  = fetchWhisperSmallWeights()
    loc_tok  = get_tokenizer(audio_transcribe_params)
    weight_dict = torch.load(loc_weight, weights_only=False)
    return weight_dict["model_state_dict"], loc_tok