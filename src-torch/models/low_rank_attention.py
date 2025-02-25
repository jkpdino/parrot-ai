import math
import torch
import torch.nn as nn

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

        self.proj_up = nn.Linear(config.rank, config.dimension)
        
        self.config = config

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b, n, d = x.shape
        if d != self.config.dimension:
            raise ValueError(f"Input dimension {d} doesn't match config dimension {self.config.dimension}")
        if mask is not None and mask.shape != (n, n):
            raise ValueError(f"Mask shape {mask.shape} doesn't match expected shape {(n, n)}")

        r = self.config.rank
        head_size = r // self.config.heads

        # Project and reshape all at once
        q = self.q_proj(x).view(b, n, self.config.heads, head_size).permute(0, 2, 1, 3).contiguous()
        k = self.k_proj(x).view(b, n, self.config.heads, head_size).permute(0, 2, 1, 3).contiguous()
        v = self.v_proj(x).view(b, n, self.config.heads, head_size).permute(0, 2, 1, 3).contiguous()

        # Combine batch and head dimensions for efficient bmm
        q = q.view(-1, n, head_size)
        k = k.view(-1, n, head_size)
        v = v.view(-1, n, head_size)

        # perform attention
        att = torch.bmm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(head_size))

        if mask is not None:
            # Expand mask for batch and heads
            mask = mask.unsqueeze(0).expand(b * self.config.heads, -1, -1)
            att = att + mask

        att = self.dropout(torch.softmax(att, dim=-1))
        y = torch.bmm(att, v)

        # Restore original shape
        y = y.view(b, self.config.heads, n, head_size)
        y = y.permute(0, 2, 1, 3).contiguous()
        y = y.view(b, n, self.config.rank)  # Changed from d to config.rank
        
        # Project back to original dimension
        y = self.proj_up(y)

        return y