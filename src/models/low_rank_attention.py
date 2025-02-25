import math
import mlx.core as mx
import mlx.nn as nn

from .config import GPTConfig

"""
Low-Rank attention is similar to a normal attention mechanism, except that it uses a low-rank approximation of the attention weights.

Parameters:
  b - Batch size
  n - Sequence length
  d - Internal dimension (config.dimension)
  h - Number of heads (config.heads)
  r - Rank of the low-rank approximation (config.rank)

Weights:
  Wq - Query matrix of shape (d, r)
  Wk - Key matrix of shape (d, r)
  Wv - Value matrix of shape (d, r)

Inputs:
  x - tensor of shape (b, n, d)
  mask - optional attention mask of shape (b, h, n, n)

Outputs:
  y - tensor of shape (b, n, d)
"""

class LowRankAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        if not hasattr(config, 'rank') or not hasattr(config, 'heads'):
            raise ValueError("Config must specify 'rank' and 'heads'")
        if config.rank % config.heads != 0:
            raise ValueError(f"Rank ({config.rank}) must be divisible by number of heads ({config.heads})")
        if config.rank > config.dimension:
            raise ValueError(f"Rank ({config.rank}) should not exceed dimension ({config.dimension})")

        self.q_proj = nn.Linear(config.dimension, config.rank)
        self.k_proj = nn.Linear(config.dimension, config.rank)
        self.v_proj = nn.Linear(config.dimension, config.rank)

        self.dropout = nn.Dropout(config.dropout)
        
        self.config = config

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        b, n, d = x.shape
        if d != self.config.dimension:
            raise ValueError(f"Input dimension {d} doesn't match config dimension {self.config.dimension}")
        if mask is not None and mask.shape != (b, self.config.heads, n, n):
            raise ValueError(f"Mask shape {mask.shape} doesn't match expected shape {(b, self.config.heads, n, n)}")

        r = self.config.rank

        # Project x to q, k, v
        # Split into heads
        # Transpose to shape (b, h, n, r//h)
        q = self.q_proj(x).reshape(b, n, self.config.heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, n, self.config.heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, n, self.config.heads, -1).transpose(0, 2, 1, 3)

        head_size = r // self.config.heads

        # perform the attention mechanism
        unscaled_att = q @ k.transpose(2, 3)
        att = unscaled_att * (1.0 / math.sqrt(head_size))

        if mask is not None:
          att = att + mask

        # perform normalization and dropout
        att = self.dropout(mx.softmax(att, axis=-1))

        # perform the weighted sum
        y = att @ v

        # transpose the heads back to the original shape
        y = y.transpose(1, 2).reshape(b, n, d)

        return y