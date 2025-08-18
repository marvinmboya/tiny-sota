import json 
from pathlib import Path 
from dataclasses import dataclass 
def load_json(path):
    with open(path, "r") as f:
        content = json.load(f)
    return content


@dataclass
class LOAD:
    root: Path = Path(__file__).parents[0]
    KOKORO_VOCAB = load_json(root / "kokoro_vocab.json")
    COCO_CLASSES = load_json(root / "coco.txt")
    IMAGENET_CLASSES = load_json(root / "imagenet.txt")
