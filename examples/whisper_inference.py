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

weights, tok = loadWhisperSmallWeightsAndTok(Audio_Transcribe_Params)
transferWhisperWeights(model, config, weights)
del weights

engine = STTEngine(model, tok, device)
speech_ops = SpeechOptions()

engine("./files/Spanish-greetings.mp3", speech_ops)
engine("./files/japanese.mp3", speech_ops)
engine.switch_task()
engine("./files/japanese.mp3", speech_ops)
engine.switch_task()
engine("./files/english.wav", speech_ops, verbose=True)






