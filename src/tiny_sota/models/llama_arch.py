import torch
import torch.nn as nn

from .tiny_modules import RMSNorm, TriFeedForward
from .attention import GQAttention
from .llm_utils import compute_rope_params
from .configs import BaseConfig


class DecoderBlock(nn.Module):
    def __init__(self, config: BaseConfig, is_qwen3=True):
        super(DecoderBlock,self).__init__()
        emb_dim = config.emb_dim 
        dtype = config.dtype
        self.rms1 = RMSNorm(emb_dim, eps=1e-5)
        self.rms2 = RMSNorm(emb_dim, eps=1e-5)
        self.attn = GQAttention(config)
        self.feed_forward = TriFeedForward(config)
    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.rms1(x)
        x = self.attn(x, mask, cos, sin) + shortcut
        shortcut = x
        x = self.rms2(x)
        x = self.feed_forward(x) + shortcut
        return x
    
class Llama3Model(nn.Module):
    def __init__(self, config: BaseConfig):
        super(Llama3Model, self).__init__()
        vocab = config.n_vocab
        context_len = config.context_len
        emb_dim = config.emb_dim
        head_dim = config.head_dim
        bias = config.bias
        dtype = config.dtype
        layers = config.layers
        self.embedding = nn.Embedding(vocab, emb_dim, dtype=dtype)
        self.decoders = nn.ModuleList([
            DecoderBlock(config) for _ in range(layers)])
        self.rms_norm = RMSNorm(emb_dim, eps=1e-5)
        self.linear = nn.Linear(emb_dim, vocab, bias=bias, dtype=dtype)
        cos, sin = compute_rope_params(head_dim, context_len, config.rope_base, config)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self,x):
        x = self.embedding(x)
        seq_len = x.shape[1]
        mask = torch.empty(seq_len, seq_len, device=x.device).fill_(-torch.inf).triu_(1)
        for decoder in self.decoders:
            x = decoder(x, mask, self.cos, self.sin)
        x = self.rms_norm(x)
        out = self.linear(x)
        return out

