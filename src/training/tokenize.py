from typing import List, Union
import torch

class LLAMATokenizer:
    def __init__(self, model_name: str = "gpt2"):
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, token_ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        return self.tokenizer.decode(token_ids)
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

def create_tokenizer(model_name: str = "gpt2") -> LLAMATokenizer:
    """Create and return a new LLAMA tokenizer instance"""
    return LLAMATokenizer(model_name)