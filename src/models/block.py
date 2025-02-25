import math
import mlx.core as mx
import mlx.nn as nn

from .config import GPTConfig
from .low_rank_attention import LowRankAttention

"""
A Transformer block implementation with Low-Rank attention and MLP.

This block implements a modified transformer architecture that uses:
- Multiple Low-Rank attention layers (controlled by attention_priority)
- Pre-layer normalization
- MLP with GELU activation
- Residual connections

Parameters:
    config (GPTConfig): Configuration object containing:
        - dimension: Model dimension
        - attention_priority: Number of attention layers
        - heads: Number of attention heads
        - dropout: Dropout probability
        - bias: Whether to use bias in linear layers
"""

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.dimension, 4 * config.dimension, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.dimension, config.dimension, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LowRankAttentionBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # Use nn.ModuleList equivalent for MLX
        self.att = nn.Sequential(*[LowRankAttention(config) 
                                 for _ in range(config.attention_priority)])
        self.att_norm = nn.Sequential(*[nn.LayerNorm(config.dimension, affine=config.bias) 
                                      for _ in range(config.attention_priority)])
        
        self.mlp = MLP(config)
        self.mlp_norm = nn.LayerNorm(config.dimension, affine=config.bias)

    def _attention_layer(self, x: mx.array, layer_idx: int, mask: mx.array = None) -> mx.array:
        """Apply a single attention layer with normalization and residual connection."""
        norm = self.att_norm[layer_idx](x)
        return x + self.att[layer_idx](norm, mask=mask)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        # Apply attention layers
        for i in range(len(self.att)):
            x = self._attention_layer(x, i, mask)

        # Apply MLP with normalization and residual connection
        x = x + self.mlp(self.mlp_norm(x))
        
        return x
