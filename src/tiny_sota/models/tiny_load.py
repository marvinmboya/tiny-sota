import torch 
from huggingface_hub import hf_hub_download
from pathlib import Path 

from tiny_sota.tiny_utils import load_model, load_weights
from tiny_sota.tiny_utils import ColorPrint

class LLMS_META:
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


def setLoadingFunc(path, opt, model=None):
    result = {
        "entire": lambda path: load_model(path),
        "dict": lambda path: load_weights(model, path)
    }[opt](path)


def getLocalWeightsDir():
    parent = Path.home()/".cache/tiny_sota"
    local_dir = parent/"models" 
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir 

def divisorWeight(value):
    i = 0
    while value:
        value = value >> 10; i += 1
    return i - 1

def getLocalWeights():
    local_dir = getLocalWeightsDir()
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

def fetchLLMWeightAndTok(meta, local_dir):
    repo_id = meta["repo_id"]
    commit = meta["commit"]
    weight_id = meta["weight_id"]
    tok_id = meta["tok_id"]
    loc_weight = meta["loc_weight"]
    loc_tok = meta["loc_tok"]
    weight_path = local_dir/loc_weight
    tok_path = local_dir/loc_tok
    if not weight_path.exists():
        hf_hub_download(repo_id, weight_id, revision=commit, local_dir=local_dir)
        (local_dir/weight_id).rename(weight_path)
    else:
        ColorPrint.Nice(f"{weight_path} exists!")
    # tokenizer
    if not tok_path.exists():
        hf_hub_download(repo_id, tok_id, revision=commit, local_dir=local_dir)
        (local_dir/tok_id).rename(tok_path)
    else:
        ColorPrint.Nice(f"{tok_path} exists!")


def assign(left, right, tensor_name="unknown", cast_to=None):
        cast_to = cast_to or right.clone().detach().dtype
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(
            right.clone().detach().to(cast_to) 
            if isinstance(right, torch.Tensor) 
            else torch.tensor(right)
        )