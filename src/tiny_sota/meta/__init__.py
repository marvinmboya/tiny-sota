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
    AMERICAN_ENGLISH = "af_heart"
    AMERICAN_ENGLISH2 = "af_bella"
    BRITISH_ENGLISH = "bf_emma"
    JAPANESE = "jf_alpha"
    HINDI = "hf_alpha"
    MANDARIN =  "zf_xiaobei"
    SPANISH = "ef_dora"
    BRAZ_PORT = "pf_dora"
    FRENCH = "ff_siwis"
    ITALIAN = "if_sara"


@dataclass(frozen=True)
class _MALE:
    AMERICAN_ENGLISH = "am_fenrir"
    AMERICAN_ENGLISH2 = "am_michael"
    AMERICAN_ENGLISH3 = "am_puck"
    BRITISH_ENGLISH = "bm_fable"
    JAPANESE = "jm_kumo"
    HINDI = "hm_omega"
    MANDARIN =  "zm_yunjian"
    SPANISH = "em_alex"
    BRAZ_PORT = "pm_alex"
    ITALIAN = "im_nicola"

@dataclass(frozen=True)
class KOKORO_VOICES:
    FEMALE: _FEMALE = _FEMALE()
    MALE: _MALE = _MALE()

_root: Path = Path(__file__).parents[0]
COCO_CLASSES = load_json(_root / "coco.txt")
IMAGENET_CLASSES = load_json(_root / "imagenet.txt")
WHISPER_LANGS = load_json(_root / "whisper_langs.json")
KOKORO_VOCAB = load_json(_root / "kokoro_vocab.json")
KOKORO_VOICES = KOKORO_VOICES()