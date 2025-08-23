import json 
from pathlib import Path 
from dataclasses import dataclass, field 
from typing import List 

def load_json(path):
    with open(path, "r") as f:
        content = json.load(f)
    return content

dataclass(frozen=True)
class _FEMALE:
    HEART = "af_heart"
    ALLOY = "af_alloy"
    AOEDE = "af_aoede"
    BELLA = "af_bella"
    JESSICA = "af_jessica"
    KORE = "af_kore"
    NICOLE = "af_nicole"
    NOVA = "af_nova"
    RIVER = "af_river"
    SARAH = "af_sarah"
    SKY = "af_sky"

@dataclass(frozen=True)
class _MALE:
    ADAM = "am_adam"
    ECHO = "am_echo"
    ERIC = "am_eric"
    FENRIR = "am_fenrir"
    LIAM = "am_liam"
    MICHAEL = "am_michael"
    ONYX = "am_onyx"
    PUCK = "am_puck"
    SANTA = "am_santa"

@dataclass(frozen=True)
class KOKORO_VOICES:
    FEMALE: _FEMALE = _FEMALE()
    MALE: _MALE = _MALE()

@dataclass(frozen=True)
class KOKORO_LANG_CODES:
    AMERICAN_ENGLISH = 'a'
    BRITISH_ENGLISH = 'b'
    SPANISH = 'e'
    FRENCH = 'f'
    HINDI = 'h'
    ITALIAN = 'i'
    BRAZILIAN_PORTUGUESE = 'p'
    JAPANESE = 'j'
    MANDARIN = 'z'

_root: Path = Path(__file__).parents[0]
COCO_CLASSES = load_json(_root / "coco.txt")
IMAGENET_CLASSES = load_json(_root / "imagenet.txt")
KOKORO_VOCAB = load_json(_root / "kokoro_vocab.json")
KOKORO_VOICES = KOKORO_VOICES()