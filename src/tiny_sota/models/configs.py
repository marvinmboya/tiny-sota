import torch 
from torch import Tensor 
from dataclasses import dataclass, field

from typing import Optional, Union, List, Iterable, Dict, Any

@dataclass
class BaseConfig:
    n_vocab: int =  50_257
    context_len: int = 1024
    emb_dim: int = 768
    heads: int = 12
    layers: int = 12
    hidden_dim: int = emb_dim
    head_dim: int = int(emb_dim // heads)
    qk_norm: bool = True
    q_bias: bool = False
    k_bias: bool = False
    v_bias: bool = False
    o_bias: bool = False
    n_kv_groups: int = 8
    rope_base: float = 1_000_000.0
    dtype: torch.dtype = torch.bfloat16
    eps: float = 1e-6

@dataclass
class Qwen3_06B:
    n_vocab =  151_936
    context_len = 40_960
    emb_dim = 1024
    heads = 16
    layers = 28
    hidden_dim = 3072
    head_dim = 128
    qk_norm = True
    bias: bool = False
    n_kv_groups = 8
    rope_base = 1_000_000.0
    dtype: torch.dtype = torch.bfloat16

class Llama_Freqs:
    factor = 32.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    
@dataclass
class Llama32_1B:
    n_vocab =  128_256
    context_len = 131_072
    emb_dim = 2048
    heads = 32
    layers = 16
    hidden_dim = 8192
    head_dim = 64
    qk_norm = False
    bias: bool = False
    n_kv_groups = 8
    rope_base = 500_000.0
    freq_config = Llama_Freqs()
    dtype: torch.dtype = torch.bfloat16

@dataclass
class Qwen_Dummy:
    n_vocab =  1_936
    context_len = 4_096
    emb_dim = 1024
    heads = 16
    layers = 28
    hidden_dim = 3072
    head_dim = 128
    qk_norm = True
    bias: bool = False
    n_kv_groups = 8
    rope_base = 1_000_000.0
    dtype: torch.dtype = torch.bfloat16

@dataclass 
class Qwen_Tok_Options:
    add_generation_prompt: bool = True,
    think_mode: bool = False
    
@dataclass
class Whisper_Tiny:
    n_vocab = 51864
    n_mels = 80
    n_audio_ctx = 1500
    n_audio_state = 384
    n_audio_head = 6
    n_audio_layer = 4
    n_text_ctx = 448
    n_text_state = 384
    n_text_head = 6
    n_text_layer = 4
    q_bias = True
    k_bias = False
    v_bias = True
    o_bias = True
    dtype = torch.float32

class Whisper_Small:
    n_mels = 80
    n_vocab = 51865
    n_audio_ctx = 1500
    n_audio_state = 768
    n_audio_head = 12
    n_audio_layer = 12
    n_text_ctx = 448
    n_text_state = 768
    n_text_head = 12
    n_text_layer = 12
    q_bias = True
    k_bias = False
    v_bias = True
    o_bias = True
    dtype = torch.float16

class ModelConfigs:
    Qwen = Qwen3_06B
    Llama = Llama32_1B
    Whisper = Whisper_Small
    Dummy = Qwen_Dummy

# SPEECHTOTEXT STARTS 
def exact_div(x, y):
    assert x % y == 0
    return x // y

@dataclass
class Audio_Mel_Params:
    SAMPLE_RATE: int = 16_000
    CHUNK_LENGTH: int = 30
    HOP_LENGTH: int = 160
    PCM_SCALE: float = 32768.0
    N_FFT: int = 400
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
    N_FRAMES = N_SAMPLES // HOP_LENGTH
    FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH

@dataclass
class Audio_PreDecode_Params:
    n_audio_ctx: int
    n_text_ctx: int
    initial_prompt: Optional[str] = None
    all_tokens: List[Any] = field(default_factory=list)
    all_segments: List[Any] = field(default_factory=list)
    prompt_reset_since: int = 0
    @property 
    def input_stride(self):
        return exact_div(
            Audio_Mel_Params.N_FRAMES, 
            self.n_audio_ctx
        )
    @property
    def time_precision(self): 
        return (
            self.input_stride * 
            Audio_Mel_Params.HOP_LENGTH / 
            Audio_Mel_Params.SAMPLE_RATE
        )
    @property 
    def remaining_prompt_length(self): 
        return self.n_text_ctx // 2 - 1
    def __post_init__(self):
        if self.initial_prompt is not None:
            self.initial_prompt_tokens = self.tokenizer.encode(
                " " + self.initial_prompt.strip()
            )
            self.all_tokens.extend(self.initial_prompt_tokens)
            self.remaining_prompt_length -= len(self.initial_prompt_tokens)
        else:
            self.initial_prompt_tokens = []

class AudioTasks:
    Translate: str = "translate"
    Transcribe: str = "transcribe"

@dataclass
class Audio_Transcribe_Params:
    language: str = "en" 
    task: AudioTasks = AudioTasks.Transcribe 
    num_languages: int = 99
    is_multilingual: bool = True

@dataclass
class SpeechOptions:
    compression_ratio_threshold = 2.4
    logprob_threshold = -1.0
    no_speech_threshold =  0.6
    cond_prev_text = True 
    
@dataclass
class DecodeOptions:
    task: AudioTasks = AudioTasks.Transcribe
    language: Optional[str] = "en"
    temperature: float = 0.0
    sample_len: Optional[int] = None
    best_of: Optional[int] = None
    beam_size: Optional[int] = None
    patience: Optional[float] = None
    length_penalty: Optional[float] = None
    prompt: Optional[Union[str, List[int]]] = None
    prefix: Optional[Union[str, List[int]]] = None
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: Optional[float] = 1.0
    dtype: torch.dtype = torch.float16

@dataclass
class DecodeResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = torch.nan
    no_speech_prob: float = torch.nan
    temperature: float = torch.nan
    compression_ratio: float = torch.nan

# TEXTTOSPEECH STARTS
@dataclass(frozen=True)
class AlbertConfig:
    n_vocab: int = 178
    emb_size: int = 128
    pos_dim: int = 512
    tok_dim: int = 2
    heads: int = 12 
    layers: int = 12
    emb_dim: int = 768
    d_in: int = 768
    d_out: int = 768
    hidden_dim: int = 2048
    q_bias: bool = True
    k_bias: bool = True
    v_bias: bool = True
    o_bias = True
    bias: bool = True
    eps: float = 1e-12
    dtype: torch.dtype = torch.float32

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}


class AudioConfigs:
    Mel_Op = Audio_Mel_Params
    Predecode_Op = Audio_PreDecode_Params
    Transcribe_Op = Audio_Transcribe_Params 
    Speech_Op = SpeechOptions 
    Decode_Op = DecodeOptions
    Decode_Res = DecodeResult