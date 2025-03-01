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
- MLP with GELU activation or Mixture of Experts (MoE)
- Residual connections

Parameters:
    config (GPTConfig): Configuration object containing:
        - dimension: Model dimension
        - attention_priority: Number of attention layers
        - heads: Number of attention heads
        - dropout: Dropout probability
        - bias: Whether to use bias in linear layers
        - use_moe: Whether to use Mixture of Experts
        - num_experts: Number of expert networks
        - num_experts_per_token: Number of experts to route each token to
        - expert_capacity_factor: Factor to determine expert capacity
        - moe_jitter_eps: Jitter for expert routing
        - moe_dropout: Dropout for expert routing
"""

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Use a single projection for up and down projections for better memory efficiency
        self.c_combined = nn.Linear(config.dimension, 4 * config.dimension + config.dimension, bias=config.bias)
        self.dropout = nn.Dropout(p=config.dropout)
        # Using nn.GELU() is more efficient than functional for repeated use
        self.gelu = nn.GELU()
        self.hidden_size = 4 * config.dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project up and down in a single operation
        combined = self.c_combined(x)
        
        # Split the combined projection
        up_proj = combined[..., :self.hidden_size]
        down_proj = combined[..., self.hidden_size:]
        
        # Apply activation
        hidden = self.gelu(up_proj)
        
        # Compute mean efficiently (avoid in-place operations for gradient checkpointing compatibility)
        mean = hidden.mean(dim=-1, keepdim=True)
        
        # Clamp mean to prevent extreme values
        mean = torch.clamp(mean, min=-1e3, max=1e3)
        
        # Multiply and apply dropout (avoid in-place operations)
        return self.dropout(down_proj * mean)

class MoEExpert(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.dimension, 4 * config.dimension, bias=config.bias)
        self.down_proj = nn.Linear(4 * config.dimension, config.dimension, bias=config.bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.gelu(self.up_proj(x))
        
        # Add gradient clipping for numerical stability
        hidden = torch.clamp(hidden, min=-1e3, max=1e3)
        
        output = self.down_proj(hidden)
        
        # Add output clipping for numerical stability
        output = torch.clamp(output, min=-1e3, max=1e3)
        
        return self.dropout(output)

class MixtureOfExperts(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Create experts
        self.experts = nn.ModuleList([MoEExpert(config) for _ in range(config.num_experts)])
        
        # Router network
        self.router = nn.Linear(config.dimension, config.num_experts, bias=False)
        
        # Dropout for routing
        self.dropout = nn.Dropout(p=config.moe_dropout)
        
        # Number of experts to route each token to
        self.num_experts_per_token = config.num_experts_per_token
        
        # Jitter for routing
        self.moe_jitter_eps = config.moe_jitter_eps
        
        # Expert capacity factor
        self.expert_capacity_factor = config.expert_capacity_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        total_tokens = batch_size * seq_len
        
        # Reshape input for routing
        x_reshaped = x.view(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)
        
        # Get router logits
        router_logits = self.router(x_reshaped)  # (batch_size * seq_len, num_experts)
        
        # Clamp router logits for numerical stability
        router_logits = torch.clamp(router_logits, min=-1e2, max=1e2)
        
        # Add noise for load balancing during training (avoid in-place operations)
        if self.training and self.moe_jitter_eps > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.moe_jitter_eps
        
        # Calculate routing probabilities with numerical stability
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Check for NaN values and replace with uniform distribution
        if torch.isnan(routing_weights).any():
            nan_mask = torch.isnan(routing_weights)
            routing_weights = torch.where(
                nan_mask, 
                torch.ones_like(routing_weights) / self.config.num_experts, 
                routing_weights
            )
            
        routing_weights = self.dropout(routing_weights)
        
        # Get top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.num_experts_per_token, dim=-1)
        
        # Normalize weights (avoid in-place operations)
        weight_sum = routing_weights.sum(dim=-1, keepdim=True)
        # Avoid division by zero
        weight_sum = torch.clamp(weight_sum, min=1e-6)
        routing_weights = routing_weights / weight_sum
        
        # Calculate expert capacity
        tokens_per_expert = int(total_tokens * self.expert_capacity_factor * self.num_experts_per_token / self.config.num_experts)
        tokens_per_expert = max(tokens_per_expert, 4)  # Minimum capacity
        
        # Initialize output tensor - reuse x_reshaped's shape but filled with zeros
        final_output = torch.zeros_like(x_reshaped)
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get indices of tokens routed to this expert
            token_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
            
            # Apply capacity constraint
            if token_indices.shape[0] > tokens_per_expert:
                # Get weights for this expert
                expert_weights = torch.zeros(total_tokens, device=x.device)
                weight_indices = (indices == expert_idx).int().argmax(dim=-1)[token_indices]
                expert_weights[token_indices] = routing_weights[token_indices, weight_indices]
                
                # Sort by routing weight and keep top tokens
                _, sorted_indices = torch.topk(expert_weights, tokens_per_expert, dim=0)
                token_indices = sorted_indices
            
            # Get tokens for this expert
            expert_inputs = x_reshaped[token_indices]
            
            # Process tokens with expert
            expert_outputs = expert(expert_inputs)
            
            # Get routing weights for these tokens to this expert
            weight_indices = (indices == expert_idx).int().argmax(dim=-1)[token_indices]
            expert_weights = routing_weights[token_indices, weight_indices].unsqueeze(-1)
            
            # Add weighted outputs to final output (use index_add instead of in-place operations)
            final_output = final_output.index_add(0, token_indices, expert_outputs * expert_weights)
        
        # Check for NaN values in final output
        if torch.isnan(final_output).any():
            final_output = torch.where(
                torch.isnan(final_output),
                x_reshaped,  # Use input as fallback
                final_output
            )
        
        # Reshape output back to original shape
        return final_output.view(batch_size, seq_len, hidden_dim)

class LowRankAttentionBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            LowRankAttention(config) for _ in range(config.attention_priority)
        ])
        
        # Layer norms for attention
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(config.dimension) for _ in range(config.attention_priority)
        ])
        
        # MLP or MoE
        if config.use_moe:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = MLP(config)
        
        # Layer norm for MLP
        self.mlp_norm = nn.LayerNorm(config.dimension)
        
    def _attention_layer(self, x: torch.Tensor, layer_idx: int, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-layer normalization
        normalized = self.attention_norms[layer_idx](x)
        
        # Apply attention
        attention_output = self.attention_layers[layer_idx](normalized, mask)
        
        # Residual connection with gradient clipping
        return x + torch.clamp(attention_output, min=-1e3, max=1e3)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Apply attention layers
        for i in range(self.config.attention_priority):
            x = self._attention_layer(x, i, mask)
            
            # Check for NaN values
            if torch.isnan(x).any():
                # Replace NaN values with zeros
                x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Apply MLP or MoE with pre-layer normalization and residual connection
        normalized = self.mlp_norm(x)
        mlp_output = self.mlp(normalized)
        
        # Clamp MLP output for stability
        mlp_output = torch.clamp(mlp_output, min=-1e3, max=1e3)
        
        return x + mlp_output
