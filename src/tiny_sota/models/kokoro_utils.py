from attr import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        assert window == 'hann', window
        self.window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))
        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
    
class CustomSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200,
        win_length=800, window="hann", center=True,
        pad_mode="replicate"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode
        self.freq_bins = self.n_fft // 2 + 1
        assert window == 'hann', window
        window_tensor = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        if self.win_length < self.n_fft:
            extra = self.n_fft - self.win_length
            window_tensor = F.pad(window_tensor, (0, extra))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[: self.n_fft]
        self.register_buffer("window", window_tensor)
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)
        forward_window = window_tensor.numpy()
        forward_real = dft_real * forward_window
        forward_imag = dft_imag * forward_window

        forward_real_torch = torch.from_numpy(forward_real).float()
        forward_imag_torch = torch.from_numpy(forward_imag).float()
        self.register_buffer(
            "weight_forward_real", forward_real_torch.unsqueeze(1)
        )
        self.register_buffer(
            "weight_forward_imag", forward_imag_torch.unsqueeze(1)
        )

        inv_scale = 1.0 / self.n_fft
        n = np.arange(self.n_fft)
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft
        idft_cos = np.cos(angle_t).T
        idft_sin = np.sin(angle_t).T

        inv_window = window_tensor.numpy() * inv_scale
        backward_real = idft_cos * inv_window
        backward_imag = idft_sin * inv_window

        self.register_buffer(
            "weight_backward_real", torch.from_numpy(backward_real).float().unsqueeze(1)
        )
        self.register_buffer(
            "weight_backward_imag", torch.from_numpy(backward_imag).float().unsqueeze(1)
        )
        
    def transform(self, waveform: torch.Tensor):
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)

        x = waveform.unsqueeze(1)
        real_out = F.conv1d(x, self.weight_forward_real,
            bias=None, stride=self.hop_length, padding=0,
        )
        # Imag part
        imag_out = F.conv1d(x, self.weight_forward_imag,
            bias=None, stride=self.hop_length, padding=0,
        )
        # magnitude, phase
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase


    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        real_part = real_part
        imag_part = imag_part

        real_rec = F.conv_transpose1d(real_part, self.weight_backward_real,
            bias=None, stride=self.hop_length, padding=0,
        )
        imag_rec = F.conv_transpose1d(
            imag_part, self.weight_backward_imag, bias=None,
            stride=self.hop_length, padding=0,
        )
        waveform = real_rec - imag_rec
        if self.center:
            pad_len = self.n_fft // 2
            waveform = waveform[..., pad_len:-pad_len]
        if length is not None:
            waveform = waveform[..., :length]

        return waveform

    def forward(self, x: torch.Tensor):
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])
    
class SineGen(nn.Module):
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        if not self.flag_for_pulse:
            rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1/self.upsample_scale, mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * torch.pi)
        return sines
    
    def forward(self, f0):
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise