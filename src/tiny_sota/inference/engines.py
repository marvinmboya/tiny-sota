import torch 
from dataclasses import dataclass
from .tokenizers import (
    Llama3Tokenizer, Qwen3Tokenizer
)
from ..tiny_utils.display import bcolors
from ..tiny_utils import get_device
from .utils import generate_text_stream

@dataclass
class TokenizerChoices:
    qwen = Qwen3Tokenizer 
    llama3 = Llama3Tokenizer

def loadTokenizer(file, tokenizer_class: TokenizerChoices, **kwargs):
    tokenizer_object = tokenizer_class(file, **kwargs)
    return tokenizer_object

def colorFlush(token, color=bcolors.NICE):
    print(f"{color}{token}{bcolors.ENDC}",end="",flush=True)

class LLMEngine():
    def __init__(self, 
        loaded_model, tokenizer_file: str, 
        tokenizer_choice:TokenizerChoices, 
        **kwargs):
        self.model = loaded_model.eval() 
        self.tokenizer = loadTokenizer(
                            tokenizer_file, 
                            tokenizer_choice, 
                            **kwargs) 
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_new_tokens = 2000
    def __call__(self, prompt):
        max_new_tokens = self.max_new_tokens
        device = get_device()
        tokens = self.tokenizer.encode(prompt)
        token_ids = torch.tensor(tokens,device=device).unsqueeze(0)
        for token in generate_text_stream(
            self.model, token_ids, max_new_tokens, 
            eos_token_id=self.eos_token_id):
            token_id = token.squeeze(0).tolist()
            colorFlush(self.tokenizer.decode(token_id))
