import torch 
from huggingface_hub import hf_hub_download
from pathlib import Path 
import requests

from tiny_sota.tiny_utils import load_model, load_weights
from tiny_sota.tiny_utils import ColorPrint

class MODELS_META:
    Qwen3_06B = {
        "repo_id": "Qwen/Qwen3-0.6B",
        "commit": "167b8104f88905a951069f5f95f9776908da5f68",
        "weight_id": "model.safetensors",
        "tok_id": "tokenizer.json",
        "loc_weight": "qwen3_06B.safetensors",
        "loc_tok": "qwen3_06B.json"
    }
    Llama32_1B = {
        "repo_id": "meta-llama/Llama-3.2-1B",
        "commit": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
        "weight_id": "model.safetensors",
        "tok_id": "tokenizer.json",
        "loc_weight": "llama32_1B.safetensors",
        "loc_tok": "llama32_1B.json"
    }
    Whisper_Small = {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        "loc_weight": "whisper_small.pt",
    }
    Kokoro_82M = {
        "repo_id": "hexgrad/Kokoro-82M",
        "commit": "41e5892b9d8b43e56fc560f892312a328a410973",
        "weight_id": "kokoro-v1_0.pth",
        "loc_weight": "kokoro_82M.pth"
    }


def setLoadingFunc(path, opt, model=None):
    result = {
        "entire": lambda path: load_model(path),
        "dict": lambda path: load_weights(model, path)
    }[opt](path)


def getLocalDir(dir: str = "models"):
    parent = Path.home()/".cache/tiny_sota"
    local_dir = parent/dir 
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir 

def divisorWeight(value):
    i = 0
    while value:
        value = value >> 10; i += 1
    return i - 1

def getLocalWeights():
    local_dir = getLocalDir()
    local_weights = list(local_dir.glob("*.safetensors"))
    for ext in ["*.pth", "*.pt"]:
        local_weights.extend(local_dir.glob(ext))
    local_weights = [Path(w) for w in local_weights]
    return local_weights

def showLocalWeights():
    local_weights = getLocalWeights()
    ColorPrint.Nice(f"{'SIZE':.^25}|{'NAME':.^25}")
    for nm in local_weights:
        size = nm.stat().st_size
        i = divisorWeight(size)
        size = f"{size / 1024**i:.2f} {'MB' if i < 3 else 'GB'}"
        ColorPrint.Nice(f"{size:.^25}|{nm.name:.^25}")

def fetchFilesHuggingFace(
        *, repo_id, commit, rem_id, loc_id, local_dir):
    local_path = local_dir/loc_id
    if not local_path.exists():
        hf_hub_download(repo_id, rem_id, revision=commit, local_dir=local_dir)
        (local_dir/rem_id).rename(local_path)
    else:
        ColorPrint.Nice(f"{local_path} exists!")
    return local_path

def fetchGenericFiles(url, local_dir, filename):
    from tqdm import tqdm
    weight_path = local_dir/filename
    if not weight_path.exists():
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(weight_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    else:
        ColorPrint.Nice(f"{weight_path} exists!")
    return weight_path

def assign(left, right, tensor_name="unknown", cast_to=None):
        cast_to = cast_to or right.clone().detach().dtype
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(
            right.clone().detach().to(cast_to) 
            if isinstance(right, torch.Tensor) 
            else torch.tensor(right)
        )