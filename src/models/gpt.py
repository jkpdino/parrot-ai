"""
inputs:
  T: mx.array - (b, n)

outputs:
  T: mx.array - (b, n, w)
"""

import math
import mlx.core as mx
import mlx.nn as nn

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
        
        # Use proper module container
        self.decoders = nn.Sequential(*[
            LowRankAttentionBlock(config) for _ in range(config.layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.dimension)
        self.unembed = nn.Linear(config.dimension, config.vocab_size)
        
        # Cache for attention
        self.cached_mask = None
        self.config = config

    def generate_token(self, tokens: mx.array, temperature: float = 1.0) -> mx.array:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
            
        # Scale logits before softmax for better numerical stability
        logits = self(tokens, use_mask=True)
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature scaling before softmax
        scaled_logits = next_token_logits / max(temperature, 1e-5)
        
        # Optional: Add top-k sampling
        # top_k = 40
        # v, _ = mx.topk(scaled_logits, min(top_k, scaled_logits.shape[-1]))
        # scaled_logits = mx.where(scaled_logits < v[:, [-1]], float('-inf'), scaled_logits)
        
        probs = mx.softmax(scaled_logits, axis=-1)
        return mx.random.categorical(probs)

    def __call__(self, T: mx.array, use_mask: bool = False):
        if T.ndim != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {T.shape}")
            
        x = self.embeddings(T)
        x = x + self.pos_embeddings(T)
        
        if use_mask:
            if self.cached_mask is None or self.cached_mask.shape[1] != T.shape[1]:
                self.cached_mask = self._create_causal_mask(T.shape[1])
            mask = self.cached_mask
        else:
            mask = None
            
        x = self.decoders(x, mask=mask)
        x = self.final_norm(x)
        x = self.unembed(x)
        
        return x

    def _create_causal_mask(self, length: int):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length)
        return mask.astype(self.embeddings.weight.dtype)

    def generate(self, 
                initial_tokens: mx.array, 
                max_length: int = 100, 
                temperature: float = 1.0,
                stop_token: int = None) -> mx.array:
        """Generate complete sequences.
        
        Args:
            initial_tokens: mx.array of shape (batch_size, initial_sequence_length)
            max_length: maximum number of tokens to generate
            temperature: sampling temperature
            stop_token: optional token ID to stop generation when encountered
            
        Returns:
            mx.array of shape (batch_size, final_sequence_length)
        """
        current_tokens = initial_tokens
        
        for _ in range(max_length):
            next_token = self.generate_token(current_tokens, temperature)
            current_tokens = mx.concatenate([current_tokens, next_token[:, None]], axis=1)[:, -self.config.max_seq_len:]
            
            # Stop if all sequences have generated the stop token
            if stop_token is not None and mx.all(next_token == stop_token):
                break
                
        return current_tokens