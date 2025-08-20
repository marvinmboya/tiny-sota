import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm 
import math 

from ..meta import KOKORO_VOCAB
from .attention import Attention
from .configs import AlbertConfig, KokoroConfig 
from .kokoro_utils import TorchSTFT, CustomSTFT, SineGen

from ..tiny_utils.core import get_device

class Encoder(nn.Module):
    def __init__(self, config: AlbertConfig):
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
        self.mlp_ln = nn.LayerNorm(config.d_out)
    def forward(self, x, mask=None):
        x = self.ff_in(x)
        for _ in range(self.config.layers):
            shortcut = x
            x = self.attn(x, mask=mask)
            x = self.attn_ln(x + shortcut)
            shortcut = x
            x = self.mlp(x)
            x = self.mlp_ln(x + shortcut)
        return x

class Albert(nn.Module):
    def __init__(self, config: AlbertConfig):
        super(Albert, self).__init__()
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
        x1 = self.encoder(x, mask)
        x2 = self.ff_out(x1[:, 0])
        x2 = self.act_out(x2)
        return x1

class AdaIN1d(nn.Module):
    def __init__(self, style_dim, n_feats):
        super().__init__()
        self.norm = nn.InstanceNorm1d(n_feats, affine=True)
        self.fc = nn.Linear(style_dim, n_feats*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class AdaINResBlock1(nn.Module):
    def __init__(self, chans, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        norm_conv = lambda i: weight_norm(
            nn.Conv1d(chans, chans, kernel_size, 1, dilation=dilation[i],
                      padding=get_padding(kernel_size, dilation[i]))
        )
        self.convs1 = nn.ModuleList([
            norm_conv(0), norm_conv(1), norm_conv(2)
        ])
        self.convs2 = nn.ModuleList([
            norm_conv(0), norm_conv(0), norm_conv(0)
        ])
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, chans),
            AdaIN1d(style_dim, chans),
            AdaIN1d(style_dim, chans),
        ])
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, chans),
            AdaIN1d(style_dim, chans),
            AdaIN1d(style_dim, chans),
        ])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, chans, 1))]*3)
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, chans, 1))]*3)

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
                self.convs1, self.convs2, 
                self.adain1, self.adain2, 
                self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)
            x = c2(xt) + x
        return x
    
class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, chans, eps=1e-5):
        super().__init__()
        self.chans = chans
        self.eps = eps
        self.fc = nn.Linear(style_dim, chans*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.chans,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)
    
class DurationEncoder(nn.Module):
    def __init__(self, style_dim, hidden_dim, layers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(layers):
            self.lstms.append(
                nn.LSTM(hidden_dim + style_dim, hidden_dim // 2, 
                    batch_first=True, bidirectional=True)
                )
            self.lstms.append(AdaLayerNorm(style_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')
        
class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample='none', dropout=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout)
        self.up_name = upsample
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.actv(self.norm2(x, s))
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2))
        return out
    
class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, hidden_dim, layers, max_duration, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(style_dim, hidden_dim, layers, dropout)
        self.lstm = nn.LSTM(
            hidden_dim + style_dim, hidden_dim // 2, 1, 
            batch_first=True, bidirectional=True
        )
        self.duration_proj = nn.Linear(hidden_dim, max_duration)
        self.shared = nn.LSTM(
            hidden_dim + style_dim, hidden_dim // 2, 
            1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([
            AdainResBlk1d(hidden_dim, hidden_dim, style_dim, dropout=dropout),
            AdainResBlk1d(hidden_dim, hidden_dim // 2, style_dim, upsample=True, dropout=dropout),
            AdainResBlk1d(hidden_dim // 2, hidden_dim // 2, style_dim, dropout=dropout)
        ])
        self.N = nn.ModuleList([
            AdainResBlk1d(hidden_dim, hidden_dim, style_dim, dropout=dropout),
            AdainResBlk1d(hidden_dim, hidden_dim // 2, style_dim, upsample=True, dropout=dropout),
            AdainResBlk1d(hidden_dim // 2, hidden_dim // 2, style_dim, dropout=dropout)
        ])
        self.F0_proj = nn.Conv1d(hidden_dim // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(hidden_dim // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        m = m.unsqueeze(1)
        lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(d, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en
    
    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
class TextEncoder(nn.Module):
    def __init__(self, chans, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, chans)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(chans, chans, kernel_size=kernel_size, padding=padding)),
                LayerNorm(chans),
                actv,
                nn.Dropout(0.2),
            ))
        self.lstm = nn.LSTM(chans, chans//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
        lengths = input_lengths if input_lengths.device == torch.device('cpu') else input_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x

class Decoder(nn.Module):
    def __init__(self, dim_in, style_dim, dim_out, 
                 resblock_kernel_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 resblock_dilation_sizes,
                 upsample_kernel_sizes,
                 gen_istft_n_fft, gen_istft_hop_size,
                 disable_complex=False):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList()
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, 
                                   upsample_initial_channel, resblock_dilation_sizes, 
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=disable_complex)

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x = self.generator(x, s, F0_curve)
        return x

class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)
        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
    
class Generator(nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=False):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
                    harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                   k, u, padding=(k-u)//2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1,3,5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = (
            CustomSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
            if disable_complex
            else TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
        )

    def forward(self, x, s, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1) 
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)
    
class Kokoro(nn.Module):
    def __init__(self, config: KokoroConfig, device=get_device()):
        super().__init__()
        self.device = device
        self.KOKORO_VOCAB = KOKORO_VOCAB
        self.config = config
        self.albert = Albert(config.albert)
        self.ff = nn.Linear(config.albert.d_in, config.hidden_dim)
        self.predictor = ProsodyPredictor(
            config.style_dim, config.hidden_dim,
            config.layers, config.max_duration,
            dropout=0.2
        )
        self.text_encoder = TextEncoder(
            config.hidden_dim, config.text_encoder_kernel_size,
            config.layers, config.n_token
        )
        self.decoder = Decoder(
            config.hidden_dim, config.style_dim,
            config.n_mels, **config.istftnet
        )
    @torch.no_grad()
    def forward_with_tokens(self,
        x,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        input_lens = torch.full(
            (x.shape[0],),  x.shape[-1], 
            device=x.device, dtype=torch.long
        )
        text_mask = torch.arange(input_lens.max()).unsqueeze(0).expand(
            input_lens.shape[0], -1).type_as(input_lens)
        text_mask = torch.gt(text_mask + 1, input_lens.unsqueeze(1)
            ).to(self.device)
        bert_dur = self.albert(x, mask=(~text_mask).int())
        d_en = self.ff(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lens, text_mask)
        xo, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(xo)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_duration = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(x.shape[1], device=self.device), 
            pred_duration
        )
        pred_aln_trg = torch.zeros(
            (x.shape[1], indices.shape[0]), 
            device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(x, input_lens, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_duration
    def forward(self, phonemes, ref_s, speed=1):
        input_ids = list(
          filter(
              lambda i: i is not None, 
              map(lambda p: self.KOKORO_VOCAB.get(p), phonemes)
          )
        )
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        return audio, pred_dur
