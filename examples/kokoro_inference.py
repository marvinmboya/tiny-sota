from tiny_sota.models.kokoro_arch import Kokoro, KokoroConfig
from tiny_sota.models.kokoro_load import loadKokoroWeightsAndVoice
from tiny_sota.transfers import transferKokoroWeights
from tiny_sota.meta import KOKORO_VOICES
from tiny_sota.inference import TTSEngine

device = "cpu"
config = KokoroConfig()
model = Kokoro(config, device=device)

weights, voice_pack = loadKokoroWeightsAndVoice(
    KOKORO_VOICES.MALE.AMERICAN_ENGLISH3
)
transferKokoroWeights(model, config, weights)
model.to(device).eval()
del weights 

engine = TTSEngine(
    model, 
    voice_pack = voice_pack,
    device = device
)

text = "No man ever steps in the same river twice, for it's not the same river and he's not the same man."
engine(text)