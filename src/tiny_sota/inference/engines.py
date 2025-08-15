import torch 
from typing import Optional

from .utils import generate_text_stream
from ..models import ModelConfigs, AudioConfigs
from .utils import TokenizerChoices, loadTokenizer, colorFlush

from ..models.whisper_utils import load_audio, log_mel_spectrogram
from ..models.whisper_decode import decode_mel_segments

class LLMEngine():
    def __init__(self, 
        loaded_model, tokenizer_file: str, 
        tokenizer_choice:TokenizerChoices, 
        **kwargs):
        self.model = loaded_model.eval() 
        self.tokenizer = loadTokenizer(
            tokenizer_file, 
            tokenizer_choice, 
            **kwargs
        ) 
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_new_tokens = 2000
    def __call__(self, prompt, device):
        max_new_tokens = self.max_new_tokens
        tokens = self.tokenizer.encode(prompt)
        token_ids = torch.tensor(tokens,device=device).unsqueeze(0)
        for token in generate_text_stream(
            self.model, token_ids, max_new_tokens, 
            eos_token_id=self.eos_token_id):
            token_id = token.squeeze(0).tolist()
            colorFlush(self.tokenizer.decode(token_id))

class STTEngine():
    def __init__(
            self, loaded_model, tokenizer, 
            initial_prompt: Optional[str] = None,
            audio_mel_options = AudioConfigs.Mel_Op(), 
            config = ModelConfigs.Whisper(),
            decode_options = AudioConfigs.Decode_Op()
        ):
        self.model = loaded_model.eval()
        self.tokenizer = tokenizer
        self.seek = 0
        self.config = config
        self.decode_options = decode_options
        self.mel_ops = audio_mel_options
        self.n_audio_ctx = config.n_audio_ctx 
        self.n_text_ctx = config.n_text_ctx 
        assert decode_options.language is not None, "set decode language!"
        self.initial_prompt = initial_prompt
        self.set_predecode_parameters()
    def __call__(self, audio_path, speech_options, device, verbose=False):
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(
            audio, 
            self.config.n_mels,
            device=device,
            padding=self.mel_ops.N_SAMPLES
        )
        content_frames = mel.shape[-1] - self.mel_ops.N_FRAMES
        tokens, segments = decode_mel_segments(
            model = self.model,
            tokenizer = self.tokenizer,
            mel = mel, 
            content_frames = content_frames, 
            predecode_params = self.predecode_ops,
            config = self.config,
            decode_options = self.decode_options, 
            speech_options = speech_options, 
            verbose = verbose,
            device = device 
        )
    def set_predecode_parameters(self):
        self.predecode_ops = AudioConfigs.Predecode_Op(
            self.n_audio_ctx, self.n_text_ctx, self.initial_prompt
        )


