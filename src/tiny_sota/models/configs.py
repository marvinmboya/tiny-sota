import torch 
from dataclasses import dataclass 

@dataclass
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

@dataclass
class Qwen3_06B:
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

class Llama_Freqs:
    factor = 32.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    
@dataclass
class Llama32_1B:
    n_vocab =  128_256
    context_len = 131_072
    emb_dim = 2048
    heads = 32
    layers = 16
    hidden_dim = 8192
    head_dim = 64
    qk_norm = False
    bias: bool = False
    n_kv_groups = 8
    rope_base = 500_000.0
    freq_config = Llama_Freqs()
    dtype: torch.dtype = torch.bfloat16

@dataclass
class Qwen_Dummy:
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
class Whisper:
    n_vocab = 51864
    n_mels = 80
    n_audio_ctx = 1500
    n_audio_state = 384
    n_audio_head = 6
    n_audio_layer = 4
    n_text_ctx = 448
    n_text_state = 384
    n_text_head = 6
    n_text_layer = 4
    bias = True
    dtype = torch.float32

class Configs:
    Qwen: BaseConfig = Qwen3_06B
    Llama: BaseConfig = Llama32_1B
    Whisper: Whisper = Whisper
    Dummy: BaseConfig = Qwen_Dummy
