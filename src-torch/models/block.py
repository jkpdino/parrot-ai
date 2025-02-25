import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.c_proj = nn.Linear(4 * config.dimension, config.dimension, bias=config.bias)
        self.dropout = nn.Dropout(p=config.dropout)
        # Using nn.GELU() is more efficient than functional for repeated use
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class LowRankAttentionBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # Using ModuleList for better efficiency and proper parameter management
        self.att = nn.ModuleList([LowRankAttention(config) 
                                 for _ in range(config.attention_priority)])
        self.att_norm = nn.ModuleList([nn.LayerNorm(config.dimension) 
                                      for _ in range(config.attention_priority)])
        
        self.mlp = MLP(config)
        self.mlp_norm = nn.LayerNorm(config.dimension)

    def _attention_layer(self, x: torch.Tensor, layer_idx: int, mask: torch.Tensor = None) -> torch.Tensor:
        """Apply a single attention layer with normalization and residual connection."""
        return x + self.att[layer_idx](self.att_norm[layer_idx](x), mask=mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Apply attention layers
        for i in range(len(self.att)):
            x = self._attention_layer(x, i, mask)

        # Apply MLP with normalization and residual connection
        x = x + self.mlp(self.mlp_norm(x))
        
        return x
