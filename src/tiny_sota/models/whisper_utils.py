
import torch
import zlib 
import sys
import numpy as np 
from torch import inf 
import torch.nn.functional as F 

from torch.distributions import Categorical
import torchaudio
from subprocess import run, CalledProcessError
from functools import lru_cache 
from pathlib import Path 
from .configs import Audio_Mel_Params

from .configs import Audio_Mel_Params
N_SAMPLES = Audio_Mel_Params.N_SAMPLES

system_encoding = sys.getdefaultencoding()
if system_encoding != "utf-8":
    def make_safe(string):
        return string.encode(system_encoding, errors="replace").decode(system_encoding)
else:
    def make_safe(string):
        return string

def load_audio(audio_path, sample_rate = Audio_Mel_Params.SAMPLE_RATE):
    cmd = ["ffmpeg","-nostdin","-threads", "0",
            "-i", audio_path, "-f", "s16le",
            "-ac", "1","-acodec", "pcm_s16le",
            "-ar", str(sample_rate),"-"]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(
                f"Failed to load audio: {e.stderr.decode()}"
        ) from e
    return torch.frombuffer(
            bytearray(out), dtype=torch.int16
    ).flatten().float() / Audio_Mel_Params.PCM_SCALE

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = Path(__file__).parents[1]/"assets/whisper/mel_filters.npz"
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
    
def log_mel_spectrogram(audio, n_mels, device, padding=0):
    N_FFT = Audio_Mel_Params.N_FFT
    HOP_LENGTH = Audio_Mel_Params.HOP_LENGTH
    audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(
            audio, N_FFT, HOP_LENGTH, 
            window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
   
def play_audio(audio):
    audio = audio.unsqueeze(-1)
    torchaudio.io.play_audio(audio, Audio_Mel_Params.SAMPLE_RATE)

def verify_options(options):
    if (options.beam_size is not None 
        and options.best_of is not None):
        raise ValueError("beam_size and best_of can't be given together")
    if options.temperature == 0:
        if options.best_of is not None:
            raise ValueError("best_of with greedy sampling (T=0) is not compatible")
    if options.patience is not None and options.beam_size is None:
        raise ValueError("patience requires beam_size to be given")
    if options.length_penalty is not None and not (
        0 <= options.length_penalty <= 1
    ):
        raise ValueError("length_penalty (alpha) should be a value between 0 and 1")
    return options

def new_segment(*, start, end, seek, tokenizer, tokens, result):
    tokens = tokens.tolist()
    text_tokens = [token for token in tokens if token < tokenizer.eot]
    return {
        "seek": seek,
        "start": start,
        "end": end,
        "text": tokenizer.decode(text_tokens),
        "tokens": tokens,
        "temperature": result.temperature,
        "avg_logprob": result.avg_logprob,
        "compression_ratio": result.compression_ratio,
        "no_speech_prob": result.no_speech_prob,
    }

def pad_or_trim(mel_segment, length: int = N_SAMPLES, *, axis: int = -1):
    assert isinstance(mel_segment, torch.Tensor), "segment should be torch Tensor!"
    if mel_segment.shape[axis] > length:
        mel_segment = mel_segment.index_select(
            dim=axis, index=torch.arange(length, device=mel_segment.device)
        )
    if mel_segment.shape[axis] < length:
        pad_widths = [(0, 0)] * mel_segment.ndim
        pad_widths[axis] = (0, length - mel_segment.shape[axis])
        mel_segment = F.pad(mel_segment, [pad for sizes in pad_widths[::-1] for pad in sizes])
    return mel_segment

def get_initial_tokens(
        tokenizer, sot_sequence, 
        sample_len, n_ctx, options):
    tokens = list(sot_sequence)
    if prefix := options.prefix:
        prefix_tokens = (
            tokenizer.encode(" " + prefix.strip())
            if isinstance(prefix, str)
            else prefix)
        if sample_len is not None:
            max_prefix_len = n_ctx // 2 - sample_len
            prefix_tokens = prefix_tokens[-max_prefix_len:]
        tokens = tokens + prefix_tokens
    if prompt := options.prompt:
        prompt_tokens = (
            tokenizer.encode(" " + prompt.strip())
            if isinstance(prompt, str)
            else prompt)
        tokens = ([tokenizer.sot_prev]
            + prompt_tokens[-(n_ctx // 2 - 1) :]
            + tokens)
    return tuple(tokens)

def get_suppress_tokens(tokenizer, options):
    suppress_tokens = options.suppress_tokens
    if isinstance(suppress_tokens, str):
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]
    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []
    else:
        assert isinstance(suppress_tokens, list), \
        "suppress_tokens must be a list"
    suppress_tokens.extend([tokenizer.transcribe,
            tokenizer.translate, tokenizer.sot,
            tokenizer.sot_prev, tokenizer.sot_lm])
    if tokenizer.no_speech is not None:
        suppress_tokens.append(tokenizer.no_speech)
    return tuple(sorted(set(suppress_tokens)))

def get_audio_features(model, mel, config, decode_dtype):
    if decode_dtype == torch.float16:
        mel = mel.half()
    if mel.shape[-2:] == (
        config.n_audio_ctx,
        config.n_audio_state):
        audio_features = mel
    else:
        audio_features = model.encoder(mel)
    return audio_features

def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

def format_timestamp(seconds, always_include_hours = False, decimal_marker = "."):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )

class SuppressBlank:
    def __init__(self, tokenizer, sample_begin):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits, tokens):
        if tokens.shape[1]==self.sample_begin:
            logits[:,self.tokenizer.encode(" ")+[self.tokenizer.eot]] = -inf

class SuppressTokens:
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits, tokens):
        logits[:, self.suppress_tokens] = -inf

class MaximumLikelihoodRanker:
    def __init__(self, length_penalty):
        self.length_penalty = length_penalty

    def rank(self, tokens, sum_logprobs):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result
        
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]

class GreedyDecoder:
    def __init__(self, temperature, eot):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens, logits, sum_logprobs):
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()
        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)
        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens, sum_logprobs):
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()

class ApplyTimestampRules:
    def __init__(self, tokenizer, sample_begin,
        max_initial_timestamp_index):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits, tokens):
        time_begin = self.tokenizer.timestamp_begin
        time_no = self.tokenizer.no_timestamps
        if time_no is not None:
            logits[:, time_no] = -inf
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin:]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = len(seq)>=1 and seq[-1]>=time_begin
            penultimate_was_timestamp = len(seq)<2 or seq[-2]>=time_begin
            if last_was_timestamp:
                if penultimate_was_timestamp:
                    logits[k, time_begin:] = -inf
                else:
                    logits[k, :self.tokenizer.eot] = -inf
            timestamps = sampled_tokens[sampled_tokens.ge(time_begin)]
            if timestamps.numel() > 0:
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, time_begin:timestamp_last] = -inf

        if tokens.shape[1] == self.sample_begin:
            logits[:, :time_begin] = -inf
            if self.max_initial_timestamp_index is not None:
                last_allowed = time_begin + \
                    self.max_initial_timestamp_index
                logits[:, last_allowed + 1 :] = -inf
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, time_begin:].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, :time_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, :time_begin] = -inf
    
class CachedInference:
    def __init__(self, model, initial_token_length: int):
        self.model = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

        key_modules = [block.attn.Wk for block in self.model.decoder.blocks]
        value_modules = [block.attn.Wv for block in self.model.decoder.blocks]
        self.kv_modules = key_modules + value_modules

    def logits(self, tokens, audio_features):
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
        if tokens.shape[-1] > self.initial_token_length:
            tokens = tokens[:, -1:]
        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []