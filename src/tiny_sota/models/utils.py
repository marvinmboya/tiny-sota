import torch 
from .configs import Qwen_Dummy_Config

def compute_based_on_freq_config(inv_freq, config):
    context_len = config.context_len
    factor = config.freq_config.factor
    low_factor = config.freq_config.low_freq_factor
    high_factor = config.freq_config.high_freq_factor
        
    low_wavelen = context_len / low_factor
    high_wavelen = context_len / high_factor
    wavelen = 2 * torch.pi / inv_freq

    inv_freq = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    s_factor = (context_len / wavelen - low_factor) / (high_factor - low_factor)
    s_inv_freq = ((1 - s_factor) * (inv_freq / factor) + s_factor * inv_freq)
    is_med_freq = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
    inv_freq = torch.where(is_med_freq, s_inv_freq, inv_freq)
    return inv_freq
    
def compute_rope_params(
    head_dim, context_length=4096, theta_base=10_000, 
    config=None, dtype=torch.float32
    ):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim
        )
    )
    if config and config.freq_config is not None:
        inv_freq = compute_based_on_freq_config(inv_freq, config)
    positions = torch.arange(context_length, dtype=dtype)
    angles = torch.outer(positions,inv_freq)
    angles = torch.cat([angles, angles], dim=1)
    cos, sin = torch.cos(angles), torch.sin(angles)
    return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, seq_len, dtype):
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(dtype), k_embed.to(dtype)

def getModelMemorySize(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            total_grads += param_size
    total_buffers = sum(buf.numel() for buf in model.buffers())
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    total_memory_gb = total_memory_bytes / (1024**3)
    return total_memory_gb
