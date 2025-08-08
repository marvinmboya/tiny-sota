import torch 
from dataclasses import dataclass
from .tokenizers import (
    Llama3Tokenizer, Qwen3Tokenizer
)
from ..tiny_utils.display import bcolors
from .utils import generate_text_stream, GenerateConfig 

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
                 loaded_model, 
                 tokenizer_file: str, tokenizer_choice:TokenizerChoices, 
                 **kwargs):
        self.model = loaded_model.eval() 
        self.tokenizer = loadTokenizer(tokenizer_file, tokenizer_choice, **kwargs) 
        self.eos_token_id = self.tokenizer.eos_token_id
    def __call__(self, prompt, g_config: GenerateConfig, is_hf_model=False):

        max_new_tokens = g_config.max_new_tokens
        context_len = g_config.context_len
        device = g_config.device
        tokens = self.tokenizer.encode(prompt)
        token_ids = torch.tensor(tokens,device=device).unsqueeze(0)
        for token in generate_text_stream(
            self.model, token_ids, max_new_tokens, 
            context_len, eos_token_id=self.eos_token_id, is_hf_model=is_hf_model):
            token_id = token.squeeze(0).tolist()
            colorFlush(self.tokenizer.decode(token_id))