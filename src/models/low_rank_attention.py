import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Use a single projection matrix for better memory efficiency
        self.qk_proj = nn.Linear(config.dimension, 2 * config.rank, bias=config.bias)
        self.v_proj = nn.Linear(config.dimension, config.dimension, bias=config.bias)  # Project to full dimension

        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(config.rank // config.heads)
        
        self.config = config
        self.head_size = config.rank // config.heads
        self.d_head = config.dimension // config.heads
        
        # Register buffers to avoid recreating tensors
        self.register_buffer('_att_buffer', None, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b, n, d = x.shape
        if d != self.config.dimension:
            raise ValueError(f"Input dimension {d} doesn't match config dimension {self.config.dimension}")
        if mask is not None and mask.shape != (n, n):
            raise ValueError(f"Mask shape {mask.shape} doesn't match expected shape {(n, n)}")

        # Project q and k together for better memory efficiency
        qk = self.qk_proj(x)
        q, k = qk.chunk(2, dim=-1)
        
        # Apply layer norm to stabilize q and k
        q_norm = torch.norm(q, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        k_norm = torch.norm(k, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        q = q / q_norm
        k = k / k_norm
        
        # Reshape for multi-head attention
        q = q.view(b, n, self.config.heads, self.head_size).transpose(1, 2)
        k = k.view(b, n, self.config.heads, self.head_size).transpose(1, 2)
        
        # Project v to full dimension and split into heads
        v = self.v_proj(x).view(b, n, self.config.heads, self.d_head).transpose(1, 2)

        # Compute attention scores - use bmm for better memory efficiency
        # Reshape tensors to combine batch and head dimensions
        q = q.reshape(b * self.config.heads, n, self.head_size)
        k = k.reshape(b * self.config.heads, n, self.head_size)
        v = v.reshape(b * self.config.heads, n, self.d_head)

        # Compute attention scores efficiently - avoid using out parameter for gradient checkpointing compatibility
        att = torch.bmm(q, k.transpose(1, 2))
        att = att * self.scale  # Scale instead of in-place scaling
        
        # Clamp attention scores to prevent extreme values
        att = torch.clamp(att, min=-1e4, max=1e4)

        if mask is not None:
            # Expand mask for batch and heads
            mask = mask.unsqueeze(0).expand(b * self.config.heads, -1, -1)
            att = att + mask  # Add instead of in-place addition

        # Apply softmax and dropout with numerical stability
        att = F.softmax(att, dim=-1)
        
        # Check for NaN values and replace with zeros
        if torch.isnan(att).any():
            att = torch.where(torch.isnan(att), torch.zeros_like(att), att)
            # Renormalize
            att_sum = att.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            att = att / att_sum
            
        att = self.dropout(att)
        
        # Apply attention to values
        y = torch.bmm(att, v)

        # Restore original shape
        y = y.view(b, self.config.heads, n, self.d_head)
        y = y.transpose(1, 2).contiguous().view(b, n, d)
        
        return y