import math
import torch
import torch.nn as nn

from .config import GPTConfig

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.dimension = config.dimension
        self.max_length = config.max_seq_len
        
        if self.dimension % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={self.dimension})")
        
        self.register_buffer('_pe', self._init_pe(), persistent=False)

    def _init_pe(self):
        position = torch.arange(self.max_length, dtype=torch.float)
        dim = torch.arange(0, self.dimension, 2, dtype=torch.float)
        
        div_term = torch.exp(dim * (-math.log(10000.0) / self.dimension))
        pe = torch.zeros(self.max_length, self.dimension)
        
        # Efficient computation using outer product
        angles = torch.outer(position, div_term)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        
        return pe

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 1:
            length = indices.size(0)
        else:
            length = indices.size(1)
            
        if length > self.max_length:
            raise ValueError(f"Input sequence length {length} exceeds maximum length {self.max_length}")
            
        pe = self._pe[:length]
        
        if indices.dim() > 1:
            pe = pe.unsqueeze(0)  # Add batch dimension
            
        return pe

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dimension: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dimension)
    
    def forward(self, length: int) -> torch.Tensor:
        positions = torch.arange(length, device=self.embedding.weight.device)
        return self.embedding(positions)