import torch
import torch.nn.functional as F 
import numpy as np 
import base64
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
# from .whisper_decode import DecodeTask
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
    filters_path = Path(__file__).parents[0]/"mel_filters.npz"
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
    print(f"{filters.shape = }")
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

@lru_cache(maxsize=None)
def get_encoding(enc_name, num_languages):
    vocab_path = Path(__file__).parents[0]/f"{enc_name}.tiktoken"
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1
    return tiktoken.Encoding(
        name=Path(vocab_path).name,
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens)

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
def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

def decode_with_fallback(
        model, 
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold, 
        *,segment,
        decode_options): 
    temperatures = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    decode_result = None
    for t in temperatures:
        kwargs = {**decode_options}
        if t > 0:
            kwargs.pop("beam_size", None)
            kwargs.pop("patience", None)
        else:
            kwargs.pop("best_of", None)
        options = DecodeOptions(**kwargs, temperature=t)
        decode_result = model.decode(segment, options)
        needs_fallback = False
        if (
            compression_ratio_threshold is not None
            and decode_result.compression_ratio > 
            compression_ratio_threshold
        ):
            needs_fallback = True
        if (
            logprob_threshold is not None
            and decode_result.avg_logprob < logprob_threshold
        ):
            needs_fallback = True
        if (
            no_speech_threshold is not None
            and decode_result.no_speech_prob > no_speech_threshold
            and logprob_threshold is not None
            and decode_result.avg_logprob < logprob_threshold
        ):
            needs_fallback = False
        if not needs_fallback:
            break
    return decode_result

def get_tokenizer(*, 
        language = None, num_languages = 99, 
        task = None, is_multilingual=True):
    if is_multilingual:
        enc_name = "multilingual"
    else:
        enc_name = "gpt2"
    encoding = get_encoding(enc_name, num_languages)
    return Tokenizer(
        encoding=encoding, 
        num_languages=num_languages, 
        language=language, task=task
    )

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
    
# @torch.no_grad()
# def invoke_decode_task(model, mel, decode_options = DecodeOptions()):
#     if single := mel.ndim == 2:
#         mel = mel.unsqueeze(0)
#     result = DecodeTask(model, decode_options).run(mel)
#     return result[0] if single else result

def transcribe(
        *, model, tokenizer, mel, 
        config,
        speech_options: SpeechOptions,
        decode_options: DecodeOptions,
        initial_prompt = None,
        verbose = False,
        ):
    n_audio_ctx = config.n_audio_ctx 
    n_text_ctx = config.n_text_ctx 
    condition_on_previous_text = speech_options.condition_on_previous_text

    dtype = torch.float16 if hasattr(decode_options, "fp16", True) else torch.float32
    if dtype == torch.float32:
        decode_options.fp16 = False
    content_frames = mel.shape[-1] - N_FRAMES
    clip_timestamps = "0"
    if isinstance(clip_timestamps, str):
        clip_timestamps = [float(ts) for ts in (
            clip_timestamps.split(",") if clip_timestamps else [])]
    seek_points = [round(ts*FRAMES_PER_SECOND) for ts in clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips = list(zip(seek_points[::2], seek_points[1::2]))
    clip_idx = 0
    seek = seek_clips[clip_idx][0]
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

    with tqdm.tqdm(
        total=content_frames, 
        unit="frames",
        disable=verbose is not False) as pbar:
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            window_end_time = float((seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE)
            segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)
            mel_segment = mel[:, seek : seek + segment_size]
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)
            decode_options.prompt = all_tokens[prompt_reset_since:]
            result = decode_with_fallback(
                model, 
                speech_options.compression_ratio_threshold,
                speech_options.logprob_threshold,
                speech_options.no_speech_threshold, 
                segment = mel_segment,
                decode_options=decode_options)
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
            timestamp_tokens = tokens.ge(tokenizer.timestamp_begin)
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
                    start_timestamp_pos = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice
                    
                if single_timestamp_ending:
                    seek += segment_size
                else:
                    last_timestamp_pos = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (
                    len(timestamps) > 0
                    and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    last_timestamp_pos = (
                        timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision
                    
                current_segments.append(
                    new_segment(
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        result=result,
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
            all_segments.extend = None([
                {"id = None": i, **segment}
                for i, segment in enumerate(current_segments, start=len(all_segments))
                ])
            all_tokens.extend(
                [token for segment in current_segments for token in segment['tokens']])
            if not condition_on_previous_text or result.temperature > 0.5:
                prompt_reset_since = len(all_tokens)
            pbar.update(min(content_frames, seek) - previous_seek)

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language)
   
def play_audio(audio):
    audio = audio.unsqueeze(-1)
    torchaudio.io.play_audio(audio, SAMPLE_RATE)
