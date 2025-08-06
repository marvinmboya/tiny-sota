import re
from pathlib import Path 

from tokenizers import Tokenizer
import tiktoken
from tiktoken.load import load_tiktoken_bpe

class Llama3Tokenizer:
    def __init__(self, file):
        mergeable = load_tiktoken_bpe(file)
        self.special = {"<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001, "<|start_header_id|>": 128006,
        "<|end_header_id|>": 128007, "<|eot_id|>": 128009}
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})
        self.tokenizer = tiktoken.Encoding(
            name=Path(file).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )
    def encode(self, text, bos=False, eos=False):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.tokenizer.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>","<|quad_start|>", 
        "<|quad_end|>","<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")
    def __init__(self, file: Path, think_mode=False,
                 apply_chat_template=True, add_generation_prompt=False):
        self.think_mode = think_mode
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.tokenizer = Tokenizer.from_file(str(file))
        self.set_specials_ids()
        self.set_eos_token()
    def encode(self, text, chat_wrapped=None):
        ids = []
        chat_wrapped = chat_wrapped or self.apply_chat_template
        if chat_wrapped:
            text = self.wrap_chat(text)
        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self.tokenizer.encode(part).ids)
        return ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    def wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.think_mode:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s
    def set_specials_ids(self):
        self._special_to_id = {t: self.tokenizer.token_to_id(t) for t in self._SPECIALS}
    def set_eos_token(self):
        eos_token = "<|im_end|>" if self.think_mode else "<|endoftext|>"
        self.eos_token_id = self._special_to_id[eos_token]
