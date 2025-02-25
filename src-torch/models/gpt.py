"""
inputs:
  T: torch.Tensor - (b, n)

outputs:
  T: torch.Tensor - (b, n, w)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .block import LowRankAttentionBlock
from .positional_embeddings import SinusoidalPositionalEmbedding

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        if not 0 < config.max_seq_len <= 32768:
            raise ValueError("max_seq_len must be positive and <= 32768")
            
        self.embeddings = nn.Embedding(config.vocab_size, config.dimension)
        self.pos_embeddings = SinusoidalPositionalEmbedding(config)
        
        self.decoders = nn.ModuleList([
            LowRankAttentionBlock(config) for _ in range(config.layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.dimension)
        self.unembed = nn.Linear(config.dimension, config.vocab_size)
        
        # Register buffer for attention mask
        self.register_buffer('cached_mask', None)
        self.config = config

    @torch.no_grad()
    def generate_token(self, tokens: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
            
        logits = self(tokens, use_mask=True)
        next_token_logits = logits[:, -1, :]
        
        scaled_logits = next_token_logits / max(temperature, 1e-5)
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def forward(self, T: torch.Tensor, use_mask: bool = False):
        if T.ndim != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {T.shape}")
            
        x = self.embeddings(T)
        x = x + self.pos_embeddings(T)
        
        if use_mask:
            if self.cached_mask is None or self.cached_mask.shape[1] != T.shape[1]:
                self.cached_mask = self._create_causal_mask(T.shape[1]).to(T.device)
            mask = self.cached_mask
        else:
            mask = None
            
        for decoder in self.decoders:
            x = decoder(x, mask=mask)
            
        x = self.final_norm(x)
        x = self.unembed(x)
        
        return x

    def _create_causal_mask(self, length: int):
        mask = torch.triu(torch.ones(length, length), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    @torch.no_grad()
    def generate(self, 
                initial_tokens: torch.Tensor, 
                max_length: int = 100, 
                temperature: float = 1.0,
                stop_token: int = None) -> torch.Tensor:
        """Generate complete sequences.
        
        Args:
            initial_tokens: torch.Tensor of shape (batch_size, initial_sequence_length)
            max_length: maximum number of tokens to generate
            temperature: sampling temperature
            stop_token: optional token ID to stop generation when encountered
            
        Returns:
            torch.Tensor of shape (batch_size, final_sequence_length)
        """
        current_tokens = initial_tokens
        
        for _ in range(max_length):
            next_token = self.generate_token(current_tokens, temperature)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)[:, -self.config.max_seq_len:]
            
            if stop_token is not None and (next_token == stop_token).all():
                break
                
        return current_tokens