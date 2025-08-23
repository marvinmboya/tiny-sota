from tiny_sota.models.kokoro_arch import Kokoro, KokoroConfig
from tiny_sota.models.kokoro_load import loadKokoroWeightsAndVoice
from tiny_sota.transfers import transferKokoroWeights
from tiny_sota.meta import KOKORO_VOICES, KOKORO_LANG_CODES
from tiny_sota.inference import TTSEngine

device = "cpu"
config = KokoroConfig()
model = Kokoro(config, device=device)

weights, voice = loadKokoroWeightsAndVoice(KOKORO_VOICES.FEMALE.HEART)
transferKokoroWeights(model, config, weights)
model.to(device).eval()
del weights 

engine = TTSEngine(
    model, 
    KOKORO_LANG_CODES.AMERICAN_ENGLISH,
    device=device, 
    voice=voice
)

text = "No man ever steps in the same river twice, for it's not the same river and he's not the same man."
engine(text)