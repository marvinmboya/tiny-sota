import re
from pathlib import Path 

from tokenizers import Tokenizer

class Llama3Tokenizer:
    SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")
    def __init__(self, file: Path):
        self.special_to_id = {"<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001, "<|start_header_id|>": 128006,
        "<|end_header_id|>": 128007, "<|eot_id|>": 128009}
        self.special_to_id.update({
            f"<|reserved_{i}|>": 128002 + i
            for i in range(256)
            if 128002 + i not in self.special_to_id.values()})
        self.tokenizer = Tokenizer.from_file(str(file))
        self.set_eos_token()
    def encode(self, text):
        ids = []
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
    def set_eos_token(self):
        eos_token = "<|end_of_text|>"
        self.eos_token_id = self.special_to_id[eos_token]