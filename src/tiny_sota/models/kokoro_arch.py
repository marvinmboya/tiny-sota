from .configs import Albert 
import torch 
import torch.nn as nn 
from .attention import Attention

class Encoder(nn.Module):
    def __init__(self, config: Albert):
        super(Encoder,self).__init__()
        self.config = config
        self.ff_in = nn.Linear(config.emb_size, config.d_in)
        self.attn = Attention(config)
        self.attn_ln = nn.LayerNorm(config.d_out, eps=config.eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_in, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.d_out)
        )
    def forward(self, x, mask=None):
        x = self.ff_in(x)
        for _ in range(self.config.layers):
            shortcut = x
            x = self.attn(x, mask=mask)
            x = self.attn_ln(x + shortcut)
            shortcut = x
            x = self.mlp(x)
            x = self.attn_ln(x + shortcut)
        return x

class AlbertModel(nn.Module):
    def __init__(self, config: Albert):
        super(AlbertModel,self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.emb_size)
        self.pos_emb = nn.Embedding(config.pos_dim, config.emb_size)
        self.tok_emb = nn.Embedding(config.tok_dim, config.emb_size)
        self.ln = nn.LayerNorm(config.emb_size, eps=config.eps)
        self.encoder = Encoder(config)
        self.ff_out = nn.Linear(config.d_out, config.d_out)
        self.act_out = nn.Tanh()
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device)
        pos_ids = pos_ids.unsqueeze(0)
        token_ids = torch.zeros_like(x)
        x = self.embedding(x)
        pos_x = self.pos_emb(pos_ids)
        tok_x = self.tok_emb(token_ids)
        x = pos_x + tok_x + x
        x = self.ln(x)
        x1 = self.encoder(x)
        x2 = self.ff_out(x1[:, 0])
        x2 = self.act_out(x2)
        return x1,x2