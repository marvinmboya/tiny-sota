from tiny_sota.inference import STTEngine
from tiny_sota.models import (
    Whisper, transferWhisperWeights, ModelConfigs
)
from tiny_sota.models.configs import SpeechOptions
from tiny_sota.models import transferWhisperWeights
from tiny_sota.models import loadWhisperSmallWeightsAndTok

from tiny_sota.models.configs import Audio_Transcribe_Params, AudioTasks
from tiny_sota.tiny_utils import get_device

device = get_device()
config = ModelConfigs.Whisper
model = Whisper(config)

transcribe_params = Audio_Transcribe_Params()
transcribe_params.language = "en"
weights, tok = loadWhisperSmallWeightsAndTok(transcribe_params)
transferWhisperWeights(model, config, weights)
del weights

engine = STTEngine(model, tok, device)
speech_ops = SpeechOptions()
engine("./files/Spanish-greetings.mp3", speech_ops)
engine("./files/japanese.mp3", speech_ops)
engine.switch_task()
engine("./files/japanese.mp3", speech_ops)
engine("./files/Spanish-greetings.mp3", speech_ops)
engine.switch_task()
engine("./files/english.wav", speech_ops, verbose=True)

transcribe_params.language = "zh" # set explicitly (minimal tool)
_, tok = loadWhisperSmallWeightsAndTok(transcribe_params)
engine = STTEngine(model, tok, device)
engine("./files/chinese.wav", speech_ops)
engine.switch_task()
engine("./files/chinese.wav", speech_ops)





