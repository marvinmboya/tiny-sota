import torch 
import soundfile as sf 

from typing import Optional, Union

from .utils import generate_text_stream, set_g2p, generate_audio
from ..models import ModelConfigs, AudioConfigs

from ..models.whisper_utils import load_audio, log_mel_spectrogram
from ..models.whisper_decode import decode_mel_segments
from ..models.configs import VoicePack

class LLMEngine():
    def __init__(self, 
        model, tokenizer: str, device: Union[torch.device, str]):
        model.to(device)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_new_tokens = 2000
    def __call__(self, prompt):
        max_new_tokens = self.max_new_tokens
        tokens = self.tokenizer.encode(prompt)
        token_ids = torch.tensor(tokens,device=self.device).unsqueeze(0)
        for token in generate_text_stream(
            self.model, token_ids, max_new_tokens, 
            eos_token_id=self.eos_token_id):
            token_id = token.squeeze(0).tolist()
            print('\x1B[38;5;216;1m' +
                self.tokenizer.decode(token_id) +
                + '\033[0m'
            )

class STTEngine():
    def __init__(
            self, model, tokenizer, device = Union[torch.device, str],
            initial_prompt: Optional[str] = None,
            audio_mel_options = AudioConfigs.Mel_Op(), 
            config = ModelConfigs.Whisper(),
            decode_options = AudioConfigs.Decode_Op()
        ):
        model.to(device),
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.decode_options = decode_options
        self.mel_ops = audio_mel_options
        self.n_audio_ctx = config.n_audio_ctx 
        self.n_text_ctx = config.n_text_ctx 
        assert decode_options.language is not None, "set decode language!"
        self.initial_prompt = initial_prompt
        self.set_predecode_parameters()
    def __call__(self, audio_path, speech_options, verbose=False):
        self.reset()
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(
            audio, 
            self.config.n_mels,
            device=self.device,
            padding=self.mel_ops.N_SAMPLES
        )
        content_frames = mel.shape[-1] - self.mel_ops.N_FRAMES
        result = decode_mel_segments(
            model = self.model,
            tokenizer = self.tokenizer,
            mel = mel, 
            content_frames = content_frames, 
            predecode_params = self.predecode_ops,
            config = self.config,
            decode_options = self.decode_options, 
            speech_options = speech_options, 
            verbose = verbose,
            device = self.device 
        )
        print(result['text'])
    def set_predecode_parameters(self):
        self.predecode_ops = AudioConfigs.Predecode_Op(
            self.n_audio_ctx, self.n_text_ctx, self.initial_prompt
        )
    def switch_task(self):
        task_id = self.tokenizer.sot_sequence[-1]
        if task_id == 50359:
            print("\033[92m task -> translate... \033[0m")
            self.tokenizer.sot_sequence = (
                self.tokenizer.sot_sequence[:-1]
                + (50358,)
            )
        if task_id == 50358:
            print("\033[92m task -> transcribe...\033[0m")
            self.tokenizer.sot_sequence = (
                self.tokenizer.sot_sequence[:-1]
                + (50359,)
            )
    def reset(self):
        self.predecode_ops.all_tokens = []
        self.predecode_ops.all_segments = []


class TTSEngine():
    def __init__(
        self, model, voice_pack: VoicePack, 
        device: Union[torch.device, str]
        ):
        self.g2p, self.lang_code = set_g2p(voice_pack.lang_code)
        self.voice = voice_pack.voice.to(device)
        self.model = model.to(device).eval()
    def __call__(self, text):
        for i, audio in enumerate(
            generate_audio(
                self.model, 
                self.g2p, 
                text, 
                self.voice, 
                self.lang_code, 
                speed=1)):
            sf.write(f'{i}.wav', audio, 24000)



