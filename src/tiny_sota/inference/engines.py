from enum import Enum
from .utils import Llama3Tokenizer, Qwen3Tokenizer

class Tokenizers(Enum):
    qwen = Qwen3Tokenizer 
    llama3 = Llama3Tokenizer

def loadTokenizer(file, tokenizer: Tokenizers):
    tokenizer = tokenizer(file)
    return tokenizer

class LLMEngine():
    def __init__(self, model_file, tokenizer_file: str, tokenizer_opt: Tokenizers):
        self.model_file = model_file 
        self.tokenizer = loadTokenizer(tokenizer_file) 
    def __call__(self, prompt):
        tokens = self.tokenizer.encode(prompt)
        words = self.tokenizer.decode(tokens)
    