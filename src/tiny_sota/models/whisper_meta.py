from dataclasses import dataclass, field
import torch 
from torch import Tensor 
from typing import List 

def exact_div(x, y):
    assert x % y == 0
    return x // y

SAMPLE_RATE = 16_000
CHUNK_LENGTH = 30
HOP_LENGTH = 160
PCM_SCALE = 32768.0
N_FFT = 400

N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH) 
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)

@dataclass
class SpeechOptions:
    compression_ratio_threshold = 2.4
    logprob_threshold = -1.0
    no_speech_threshold =  0.6
    condition_on_previous_text = True 
    
@dataclass(frozen=True)
class DecodeOptions:
    task = "transcribe"
    language = None
    temperature = 0.0
    sample_len = None
    best_of = None
    beam_size = None
    patience = None
    length_penalty = None
    prompt = None
    prefix = None
    suppress_tokens = "-1"
    suppress_blank = True
    without_timestamps = False
    max_initial_timestamp = 1.0
    fp16 = True

@dataclass(frozen=True)
class DecodeResult:
    audio_features: Tensor
    language: str
    language_probs = None
    tokens: List[int] = field(default_factory=list)
    text = ""
    avg_logprob = torch.nan
    no_speech_prob = torch.nan
    temperature = torch.nan
    compression_ratio = torch.nan

class LogitFilter:
    def apply(self, logits, tokens):
        """Apply any filtering or masking to logits in-place
        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError

class SequenceRanker:
    def rank(self, tokens, sum_logprobs):
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError
            

class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(self, tokens, logits, sum_logprobs):
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError
    
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

