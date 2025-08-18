import torch 
import torch.nn as nn 
from transformers import AlbertConfig, AlbertModel
from tiny_sota.models import Attention
from tiny_sota.models.configs import Albert
from tiny_sota.models.kokoro_load import transferKokoroWeights
from dataclasses import dataclass

config = {"hidden_size": 768,"num_attention_heads": 12,
    "intermediate_size": 2048,"max_position_embeddings": 512,
    "num_hidden_layers": 12,"dropout": 0.1,"vocab_size": 178
}



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

class CustomAlbert(nn.Module):
    def __init__(self, config: Albert):
        super(CustomAlbert,self).__init__()
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

alconf = AlbertConfig(**config)
model = AlbertModel(alconf)

def func(model, input, output):
    pass
model.pooler_activation.register_forward_hook(func)

x = torch.randint(178, (1, 178), dtype=torch.long)
out1 = model(x).last_hidden_state

config = Albert()
customalbert = CustomAlbert(config)
transferKokoroWeights(customalbert, config, model)
out2, _ = customalbert(x)
print(out1.shape)
print(out2.shape)
print(torch.allclose(out1, out2, atol=1e-2))
# print(model)
# end compare the two

# def infer(model, x):
#     model.eval()
#     out = model(x)
#     return out.last_hidden_state

# y = infer(model, x)
