import math
import mlx.core as mx
import mlx.nn as nn

from .config import GPTConfig

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.dimension = config.dimension
        self.max_length = config.max_seq_len
        
        if self.dimension % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={dimension})")
        
        # Pre-compute position encodings (not a trainable parameter)
        # Move this computation to __call__ to avoid saving in checkpoints
        self._init_pe()

    def _init_pe(self):
        position = mx.arange(self.max_length)
        dim = mx.arange(0, self.dimension, 2)
        
        div_term = mx.exp(dim * (-math.log(10000.0) / self.dimension))
        pos_expanded = mx.expand_dims(position, 1)
        pe = mx.zeros((self.max_length, self.dimension))
        
        angles = pos_expanded * div_term
        pe = pe.at[:, 0::2].add(mx.sin(angles))
        pe = pe.at[:, 1::2].add(mx.cos(angles))
        
        # Store as non-parameter attribute
        self._pe = mx.stop_gradient(pe)

    def __call__(self, indices: mx.array) -> mx.array:
        # Initialize PE if not already done (handles loading from checkpoint)
        if not hasattr(self, '_pe'):
            self._init_pe()
            
        if indices.ndim == 1:
            length = indices.shape[0]
        else:
            length = indices.shape[1]
            
        if length > self.max_length:
            raise ValueError(f"Input sequence length {length} exceeds maximum length {self.max_length}")
            
        pe = mx.stop_gradient(self._pe[:length])
        
        if indices.ndim > 1:
            pe = mx.expand_dims(pe, 0)  # Add batch dimension
            
        return pe

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dimension: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dimension)
    
    def __call__(self, length: int) -> mx.array:
        positions = mx.arange(length)
        return self.embedding(positions)