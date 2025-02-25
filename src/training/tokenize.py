from typing import List, Union

class LLAMATokenizer:
    def __init__(self, model_name: str = "gpt2"):
        # Lazy import to avoid circular dependencies
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 doesn't have pad token by default
        
    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Encode text into token IDs"""
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids)
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size"""
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Get the pad token ID"""
        return self.tokenizer.pad_token_id

def create_tokenizer(model_name: str = "gpt2") -> LLAMATokenizer:
    """Create and return a new LLAMA tokenizer instance"""
    return LLAMATokenizer(model_name)