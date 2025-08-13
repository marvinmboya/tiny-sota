import torch
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import apply_rotary_pos_emb
from .configs import BaseConfig
from .tiny_modules import RMSNorm

class Attention(nn.Module):
    def __init__(self, config: BaseConfig):
        super(Attention,self).__init__()
        """
        Let B = Batch, S = Seq_Len, H = Heads, HD = Head_Dim, D = HD * H 
        Q, K, V => (B,S,D)
        Q, K, V => (B,S,H,HD)
        Q, K, V => (B,H,S,HD)
        IR = Q @ K.T => (B,H,S,HD) @ (B,H,HD,S) => (B,H,S,S)
        IR = IR / (HD**0.5)
        IR = softmax(IR)
        VECS = IR @ V => (B,H,S,S) @ (B,H,S,HD) => (B,H,S,HD) 
        VECS => (B,S,H,HD)
        VECS => (B,S,D)
        OUT = VECS @ WO.T => (B,S,D)
        """
        assert config.emb_dim % config.heads == 0, "dim must be divisible by number of heads"
        emb_dim = config.emb_dim
        d_out = config.d_out
        self.heads = config.heads 
        bias = config.bias 
        dtype = config.dtype
        self.Wq = nn.Linear(emb_dim, d_out, bias=bias, dtype=dtype)
        self.Wk = nn.Linear(emb_dim, d_out, bias=False, dtype=dtype)
        self.Wv = nn.Linear(emb_dim, d_out, bias=bias, dtype=dtype)
        self.Wo = nn.Linear(d_out, emb_dim, bias=bias, dtype=dtype)

    def forward(self, x, mask=None, cos=None, 
                sin=None, enc_mel=None):
        B, seq_len, dim = x.shape
        dtype = x.dtype
        Q = self.Wq(x)
        K = self.Wk(x if enc_mel is None else enc_mel)
        V = self.Wv(x if enc_mel is None else enc_mel)
        Q = Q.view(*Q.shape[:2], self.heads, -1)
        K = K.view(*K.shape[:2], self.heads, -1)
        V = V.view(*V.shape[:2],self.heads, -1)
        if cos is not None:
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, seq_len, dtype)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        scores = Q @ K.transpose(-1,-2)
        if mask is not None:
            scores += mask[:seq_len,:seq_len]
        weights = F.softmax(scores/K.shape[-1]**.5,dim=-1)
        vectors = weights @ V 
        vectors = vectors.transpose(1,2)
        vectors = vectors.contiguous().view(B,seq_len,dim)
        out = self.Wo(vectors)
        return out


class GQAttention(nn.Module):
    def __init__(self, config: BaseConfig, is_qwen3=False):
        super(GQAttention,self).__init__()
        self.heads = config.heads 
        self.head_dim = config.head_dim
        self.n_kv_groups = config.n_kv_groups
        d_in = config.emb_dim
        self.d_out = self.heads * self.head_dim
        bias = config.bias
        self.group_size = self.heads // self.n_kv_groups
        dtype = config.dtype
        self.qk_norm = config.qk_norm
        assert self.heads % self.n_kv_groups == 0, "heads must be divisible by num_kv_groups"
        if not self.head_dim:
            assert d_in % self.heads == 0, "dim must be divisible by number of heads"
            self.head_dim = d_in // self.heads
        self.Wq = nn.Linear(d_in, self.d_out, bias=bias, dtype=dtype)
        self.Wk = nn.Linear(d_in, self.n_kv_groups * self.head_dim, bias=bias, dtype=dtype)
        self.Wv = nn.Linear(d_in, self.n_kv_groups * self.head_dim, bias=bias, dtype=dtype)
        self.Wo = nn.Linear(self.d_out, d_in, bias=bias, dtype=dtype)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, is_qwen3=is_qwen3)
            self.k_norm = RMSNorm(self.head_dim, is_qwen3=is_qwen3)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        B, seq_len, _ = x.shape
        dtype = x.dtype
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        Q = Q.view(B,seq_len,self.heads,self.head_dim)
        K = K.view(B,seq_len,self.n_kv_groups,self.head_dim)
        V = V.view(B,seq_len,self.n_kv_groups,self.head_dim)
        Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)
        Q = self.q_norm(Q) if self.q_norm else Q
        K = self.k_norm(K) if self.k_norm else K
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, seq_len, dtype)
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        scores = Q @ K.transpose(-1,-2)
        scores = scores + mask[:seq_len,:seq_len]
        weights = F.softmax(scores/self.head_dim**.5,dim=-1)
        vectors = weights @ V 
        vectors = vectors.transpose(1,2)
        vectors = vectors.contiguous().view(B,seq_len,self.d_out)
        out = self.Wo(vectors)
        return out  
