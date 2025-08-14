import torch
import torch.nn as nn 
import torch.nn.functional as F

from math import log

from .attention import Attention as MultiHeadAttention

def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    length = torch.arange(length)
    scaled_time = torch.outer(length, inv_timescales)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class DecoderBlock(nn.Module):
    def __init__(self, config, cross_attention=False):
        super(DecoderBlock,self).__init__()
        n_state = config.n_state
        config.emb_dim = config.n_state
        config.d_out = config.n_state
        # self attention
        self.attn = MultiHeadAttention(config)
        self.ln = LayerNorm(n_state)
        # cross attention
        self.cross_attn = (
        MultiHeadAttention(config) 
        if cross_attention else None)
        self.cross_ln = LayerNorm(n_state) if cross_attention else None
        # mlp 
        self.feed_forward = nn.Sequential(
            nn.Linear(n_state, n_state * 4),
            nn.GELU(),
            nn.Linear(n_state * 4, n_state)
        )
        self.ff_ln = LayerNorm(n_state) 
    def forward(self, x, enc_mel=None, mask=None, kv_cache=None):
        shortcut = x
        x = self.ln(x)
        x = self.attn(x, mask, kv_cache=kv_cache)
        x += shortcut

        if self.cross_attn:
            shortcut = x
            x = self.cross_ln(x)
            x = self.cross_attn(x, enc_mel=enc_mel, kv_cache=kv_cache)
            x = x + shortcut
        shortcut = x
        x = self.ff_ln(x)
        x = self.feed_forward(x) + shortcut
        return x
        

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder,self).__init__()
        n_mels = config.n_mels 
        n_ctx = config.n_audio_ctx
        n_state = config.n_audio_state
        config.heads = config.n_audio_head
        self.conv1 = nn.Conv1d(n_mels, n_state, 3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, 3, 2, 1)
        pos_emb = sinusoids(n_ctx, n_state) 
        self.register_buffer("pos_emb", pos_emb)
        config.n_state = n_state
        self.blocks = nn.ModuleList([
            DecoderBlock(config) 
            for _ in range(config.n_audio_layer)                                
        ])
        self.ln = LayerNorm(n_state)
         
    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.pos_emb).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return x 

class TextDecoder(nn.Module):
    def __init__(self, config):
        super(TextDecoder,self).__init__()
        n_vocab = config.n_vocab 
        n_ctx = config.n_text_ctx 
        n_state = config.n_text_state 
        config.heads = config.n_text_head
        self.token_emb = nn.Embedding(n_vocab, n_state)
        self.pos_emb = nn.Parameter(torch.empty(n_ctx, n_state))
        config.n_state = n_state
        self.blocks = nn.ModuleList([
            DecoderBlock(config,cross_attention=True) 
            for _ in range(config.n_text_layer)                                
        ])
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx,n_ctx).fill_(-torch.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        
    def forward(self, x, enc_mel, kv_cache=None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_emb(x) + self.pos_emb[offset: offset + x.shape[-1]]
        x = x.to(enc_mel.dtype)
        for block in self.blocks:
            x = block(x, enc_mel, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        token_emb = self.token_emb.weight.to(x.dtype)
        logits = (x @ token_emb.transpose(0,1)).float()
        return logits
    
class Whisper(nn.Module):
    def __init__(self, config):
        super(Whisper,self).__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = TextDecoder(config) 
         
    def forward(self, x, mels):
        encoded_mels = self.encoder(mels)
        return self.decoder(x, encoded_mels)
    
