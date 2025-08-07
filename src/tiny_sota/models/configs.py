import torch 
from dataclasses import dataclass 

@dataclass(frozen=True)
class BaseConfig:
    n_vocab: int =  50_257
    context_len: int = 1024
    emb_dim: int = 768
    heads: int = 12
    layers: int = 12
    hidden_dim: int = emb_dim
    head_dim: int = int(emb_dim // heads)
    qk_norm: bool = True
    bias: bool = False
    n_kv_groups: int = 8
    rope_base: float = 1_000_000.0
    dtype: torch.dtype = torch.bfloat16
    eps: float = 1e-6

@dataclass(frozen=True)
class Qwen3_06B_Config(BaseConfig):
    n_vocab =  151_936
    context_len = 40_960
    emb_dim = 1024
    heads = 16
    layers = 28
    hidden_dim = 3072
    head_dim = 128
    qk_norm = True
    bias: bool = False
    n_kv_groups = 8
    rope_base = 1_000_000.0
    dtype: torch.dtype = torch.bfloat16

@dataclass(frozen=True)
class Qwen_Dummy_Config(BaseConfig):
    n_vocab =  1_936
    context_len = 4_096
    emb_dim = 1024
    heads = 16
    layers = 28
    hidden_dim = 3072
    head_dim = 128
    qk_norm = True
    bias: bool = False
    n_kv_groups = 8
    rope_base = 1_000_000.0
    dtype: torch.dtype = torch.bfloat16

@dataclass
class Configs:
    Qwen: BaseConfig = Qwen3_06B_Config
    Dummy: BaseConfig = Qwen_Dummy_Config
