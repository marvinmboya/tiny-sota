import re
from pathlib import Path 

from tokenizers import Tokenizer

class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>","<|quad_start|>", 
        "<|quad_end|>","<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")
    def __init__(self, file: Path, **kwargs):
        self.think_mode = kwargs.get('think_mode', False)
        self.apply_chat_template = kwargs.get('apply_chat_template', True)
        self.add_generation_prompt = kwargs.get('add_generation_prompt',False)
        self.tokenizer = Tokenizer.from_file(str(file))
        self.set_specials_ids()
        self.set_eos_token()
    def encode(self, text, chat_wrapped=None):
        ids = []
        chat_wrapped = chat_wrapped or self.apply_chat_template
        if chat_wrapped:
            text = self.wrap_chat(text)
        stripped = text.strip()
        if stripped in self.special_to_id and "\n" not in stripped:
            return [self.special_to_id[stripped]]
        for part in filter(None, self.SPLIT_RE.split(text)):
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
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
        self.special_to_id = {t: self.tokenizer.token_to_id(t) for t in self._SPECIALS}
    def set_eos_token(self):
        eos_token = "<|im_end|>" if self.think_mode else "<|endoftext|>"
        self.eos_token_id = self.special_to_id[eos_token]
