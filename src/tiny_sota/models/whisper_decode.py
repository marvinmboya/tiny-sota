
import torch 
import torch.nn as nn
from torch import Tensor, device

from .configs import DecodeOptions, SpeechOptions
from .whisper_utils import pad_or_trim, make_safe, new_segment, format_timestamp
from .configs import Audio_Mel_Params, Audio_PreDecode_Params, Whisper_Small
from .whisper_decode_inner import DecodeTask

@torch.no_grad()
def invoke_decode_task(model, tokenizer, mel, config, decode_options):
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)
    result = DecodeTask(model, tokenizer, config, decode_options).run(mel)
    return result[0] if single else result

def decode_mel_segments(
    *,
    model: nn.Module,
    tokenizer,
    mel: Tensor, 
    content_frames: int, 
    predecode_params: Audio_PreDecode_Params,
    config: Whisper_Small,
    decode_options: DecodeOptions, 
    speech_options: SpeechOptions, 
    verbose: bool,
    device: device):
    SAMPLE_RATE = Audio_Mel_Params.SAMPLE_RATE
    HOP_LENGTH = Audio_Mel_Params.HOP_LENGTH
    N_FRAMES = Audio_Mel_Params.N_FRAMES
    N_FRAMES = Audio_Mel_Params.N_FRAMES
    all_tokens = predecode_params.all_tokens
    all_segments = predecode_params.all_segments
    input_stride = predecode_params.input_stride
    time_precision = predecode_params.time_precision
    initial_prompt_tokens = predecode_params.initial_prompt_tokens
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

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=decode_options.language,
    )


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
