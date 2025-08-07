import torch, torch.nn as nn 
import torch.nn.functional as F

from .tiny_modules import RMSNorm, TriFeedForward
from .attention import GQAttention
from .utils import compute_rope_params
from .configs import BaseConfig

class DecoderBlock(nn.Module):
    def __init__(self, config: BaseConfig):
        super(DecoderBlock,self).__init__()
        emb_dim = config.emb_dim 
        dtype = config.dtype
        self.rms1 = RMSNorm(emb_dim, dtype=dtype)
        self.rms2 = RMSNorm(emb_dim, dtype=dtype)
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

class Qwen3Model(nn.Module):
    def __init__(self, config: BaseConfig):
        super(Qwen3Model, self).__init__()
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
        self.rms_norm = RMSNorm(emb_dim, dtype=dtype)
        self.linear = nn.Linear(emb_dim, vocab, bias=bias, dtype=dtype)
        mask = torch.triu(torch.ones(context_len,context_len), diagonal=1)
        cos, sin = compute_rope_params(head_dim, context_len, config.rope_base)
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self,x):
        x = self.embedding(x)
        for decoder in self.decoders:
            x = decoder(x, self.mask, self.cos, self.sin)
        x = self.rms_norm(x)
        out = self.linear(x)
        return out


