import torch, torch.nn as nn 
import torch.nn.functional as F 

from .configs import BaseConfig 

class TriFeedForward(nn.Module):
    def __init__(self, config: BaseConfig):
        super(TriFeedForward,self).__init__()
        emb_dim = config.emb_dim
        hidden_dim = config.hidden_dim
        bias = config.bias
        dtype = config.dtype
        self.w1 = nn.Linear(emb_dim, hidden_dim, bias=bias, dtype=dtype)
        self.v = nn.Linear(emb_dim, hidden_dim, bias=bias, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, emb_dim, bias=bias, dtype=dtype)
    def forward(self,x):
        x = F.silu(self.w1(x)) * self.v(x)
        x = self.w2(x)
        return x 
    
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, bias=False, is_qwen3=False, eps=1e-6):
        super(RMSNorm,self).__init__()
        self.is_qwen3 = is_qwen3
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = (nn.Parameter(torch.zeros(emb_dim)) 
                    if bias else None)
    def forward(self,x):
        input_dtype = x.dtype
        if self.is_qwen3:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(self.eps + variance)
        rms_norm = norm_x * self.weight
        if self.bias is not None:
            rms_norm += rms_norm + self.bias
        return rms_norm.to(input_dtype)