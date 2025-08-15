import tiktoken 
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path 
import base64

from typing import List, Optional, Tuple, Dict  
from ..models.configs import Audio_Transcribe_Params, LANGUAGES

@dataclass
class Tokenizer:
    encoding: tiktoken.Encoding
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.encoding.encode(symbol),self.encoding.encode(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])
        return tuple(sorted(result))

@lru_cache(maxsize=None)
def get_encoding(enc_name, num_languages):
    vocab_path = Path(__file__).parents[1]/f"assets/whisper/{enc_name}.tiktoken"
    ranks = {base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)}
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


def get_tokenizer(audio_transcribe_params: Audio_Transcribe_Params):
    task = audio_transcribe_params.task
    language = audio_transcribe_params.language
    is_multilingual = audio_transcribe_params.is_multilingual
    num_languages = audio_transcribe_params.num_languages
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