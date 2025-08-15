import torch
import torch.nn.functional as F 
import numpy as np 
import tiktoken 
import tqdm 
import sys 

import torchaudio
from subprocess import run, CalledProcessError
from functools import lru_cache 
from pathlib import Path 

from ..tiny_utils import get_device 
from .whisper_meta import (
    LANGUAGES, SpeechOptions, DecodeOptions,
    DecodeResult, SAMPLE_RATE, 
    HOP_LENGTH, CHUNK_LENGTH, 
    PCM_SCALE, N_FFT, N_SAMPLES,
    N_FRAMES, FRAMES_PER_SECOND,
    exact_div
)

from .whisper_decode import DecodeTask
from .whisper_tok import Tokenizer

def load_audio(audio_path, sample_rate = SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", audio_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(
                f"Failed to load audio: {e.stderr.decode()}"
        ) from e
    return torch.frombuffer(
            bytearray(out), dtype=torch.int16
    ).flatten().float() / PCM_SCALE

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = Path(__file__).parents[1]/"assets/whisper/mel_filters.npz"
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
    
def log_mel_spectrogram(audio, n_mels, padding=0):
    device = get_device()
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

@torch.no_grad()
def invoke_decode_task(model, tokenizer, mel, config, decode_options):
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)
    result = DecodeTask(model, tokenizer, config, decode_options).run(mel)
    return result[0] if single else result

def decode_with_fallback(
        model, tokenizer, config, 
        compression_ratio_threshold,
        logprob_threshold, no_speech_threshold, 
        *, segment, decode_options
    ): 
    temperatures = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    decode_result = None
    for t in temperatures:
        if t > 0:
            decode_options.beam_size = None
            decode_options.patience = None
        else:
            decode_options.best_of = None
        decode_options.temperature=t
        decode_result = invoke_decode_task(model, tokenizer, segment, config, decode_options)
        needs_fallback = False
        if (compression_ratio_threshold is not None and 
            decode_result.compression_ratio > compression_ratio_threshold):
            needs_fallback = True
        if (logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold):
            needs_fallback = True
        if (no_speech_threshold is not None and decode_result.no_speech_prob > no_speech_threshold
            and logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold):
            needs_fallback = False
        if not needs_fallback:
            break
    return decode_result

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

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":
    def make_safe(string):
        return string.encode(system_encoding, errors="replace").decode(system_encoding)
else:
    def make_safe(string):
        return string
    
def transcribe(
        *, model, tokenizer, mel, 
        config,
        speech_options: SpeechOptions,
        decode_options: DecodeOptions,
        initial_prompt = None,
        verbose = False,
        ):
    seek = 0
    n_audio_ctx = config.n_audio_ctx 
    n_text_ctx = config.n_text_ctx 
    assert decode_options.language is not None, "set language!"
    device = get_device()
    content_frames = mel.shape[-1] - N_FRAMES
    input_stride = exact_div(N_FRAMES, n_audio_ctx)
    time_precision = (input_stride * HOP_LENGTH / SAMPLE_RATE)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0
    remaining_prompt_length = n_text_ctx // 2 - 1
    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
        remaining_prompt_length -= len(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []
    with tqdm.tqdm(total=content_frames, unit="frames",disable=verbose is not False) as pbar:
        while True:
            """
            this loop goes through the audio split as segments, 
            and only ends after we have decoded upto the last 
            (not so same size) segment and hence exhaust all the frames
            """
            if seek >= content_frames:
                break
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment_size = min(N_FRAMES, content_frames-seek)
            mel_segment = mel[:, seek : seek+segment_size]
            segment_duration = segment_size*HOP_LENGTH/SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device).to(decode_options.dtype)
            decode_options.prompt = all_tokens[prompt_reset_since:]
            result = decode_with_fallback(
                model, tokenizer, config,
                speech_options.compression_ratio_threshold,
                speech_options.logprob_threshold,
                speech_options.no_speech_threshold, 
                segment = mel_segment,
                decode_options=decode_options
            )
            tokens = torch.tensor(result.tokens)
            if speech_options.no_speech_threshold is not None:
                should_skip = result.no_speech_prob > speech_options.no_speech_threshold
                if (
                    speech_options.logprob_threshold is not None
                    and result.avg_logprob > speech_options.logprob_threshold
                ):
                    should_skip = False
                if should_skip:
                    seek += segment_size  
                    continue
            previous_seek = seek
            current_segments = []
            time_begin = tokenizer.timestamp_begin
            timestamp_tokens = tokens.ge(time_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)
            if len(consecutive) > 0:
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))
                last_slice = 0 
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (sliced_tokens[0].item() - time_begin)
                    end_timestamp_pos = (sliced_tokens[-1].item() - time_begin)
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            seek=seek, tokenizer=tokenizer, tokens=sliced_tokens, result=result
                        )
                    )
                    last_slice = current_slice
                if single_timestamp_ending:
                    seek += segment_size
                else:
                    last_timestamp_pos = (tokens[last_slice-1].item()-time_begin)
                    seek += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (len(timestamps) > 0 and timestamps[-1].item() != time_begin):
                    last_timestamp_pos = (timestamps[-1].item() - time_begin)
                    duration = last_timestamp_pos * time_precision 
                current_segments.append(
                    new_segment(
                        start=time_offset, end=time_offset+duration, seek=seek, 
                        tokenizer=tokenizer, tokens=tokens, result=result
                    )
                )
                seek += segment_size
            if verbose:
                for segment in current_segments:
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                    print(make_safe(line))

            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []
            all_segments.extend([
                {"id": i, **segment}
                for i, segment in enumerate(current_segments, start=len(all_segments))
                ])
            all_tokens.extend(
                [token for segment in current_segments for token in segment['tokens']])
            if not speech_options.cond_prev_text or result.temperature > 0.5:
                prompt_reset_since = len(all_tokens)
            pbar.update(min(content_frames, seek) - previous_seek)

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=decode_options.language)
   
def play_audio(audio):
    audio = audio.unsqueeze(-1)
    torchaudio.io.play_audio(audio, SAMPLE_RATE)
