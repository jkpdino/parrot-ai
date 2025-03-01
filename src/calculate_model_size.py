from models.config import GPTConfig
import argparse
from models.gpt import GPT

import torch

def calculate_model_size(config: GPTConfig):
    # Embedding layers
    embedding_params = config.vocab_size * config.dimension  # Token embeddings
    
    # Low Rank Attention parameters (per attention block)
    # In the actual implementation:
    # - qk_proj: dimension -> 2*rank (for Q and K together)
    # - v_proj: dimension -> dimension (full dimension for V)
    attention_params = (
        # qk_proj: dimension -> 2*rank
        (config.dimension * (2 * config.rank)) +
        # v_proj: dimension -> dimension
        (config.dimension * config.dimension)
    )
    if config.bias:
        attention_params += (2 * config.rank) + config.dimension  # Bias terms
    
    # MLP or MoE parameters
    if config.use_moe:
        # Router network: dimension -> num_experts
        router_params = config.dimension * config.num_experts
        
        # Each expert has up_proj and down_proj
        expert_params = config.num_experts * (
            # up_proj: dimension -> 4*dimension
            (config.dimension * (4 * config.dimension)) +
            # down_proj: 4*dimension -> dimension
            ((4 * config.dimension) * config.dimension)
        )
        
        # Add bias terms if needed
        if config.bias:
            expert_params += config.num_experts * (
                (4 * config.dimension) +  # up_proj bias
                config.dimension  # down_proj bias
            )
        
        mlp_params = router_params + expert_params
    else:
        # Standard MLP parameters
        # In the actual implementation:
        # - c_combined: dimension -> 4*dimension + dimension (combined up and down projections)
        mlp_params = config.dimension * (4 * config.dimension + config.dimension)
        if config.bias:
            mlp_params += (4 * config.dimension + config.dimension)  # Bias terms
    
    # Layer norm parameters (2 per attention block + 1 for MLP)
    layer_norm_params = 2 * config.dimension if config.bias else config.dimension
    
    # Total parameters per transformer block
    block_params = (
        (attention_params + layer_norm_params) * config.attention_priority +  # Attention blocks
        mlp_params + layer_norm_params  # MLP block
    )
    
    # Final layer norm
    final_norm_params = 2 * config.dimension if config.bias else config.dimension
    
    # Final unembedding layer
    unembed_params = config.dimension * config.vocab_size
    if config.bias:
        unembed_params += config.vocab_size
    
    # Total parameters
    total_params = (
        embedding_params +
        (block_params * config.layers) +
        final_norm_params +
        unembed_params
    )
    
    return total_params

def calculate_pytorch_model_size(config: GPTConfig):
    """Calculate model size by creating a PyTorch model instance and counting parameters"""
    model = GPT(config)
    return sum(p.numel() for p in model.parameters())

def estimate_training_memory(config: GPTConfig, batch_size: int, precision: str = 'float32', gradient_accumulation_steps: int = 1):
    """
    Estimate the GPU memory required for training the model.
    
    Args:
        config: Model configuration
        batch_size: Training batch size
        precision: Model precision ('float32', 'float16', 'bfloat16')
        gradient_accumulation_steps: Number of gradient accumulation steps
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Get parameter count
    param_count = calculate_model_size(config)
    
    # Determine bytes per parameter based on precision
    if precision == 'float32':
        bytes_per_param = 4
    elif precision in ['float16', 'bfloat16']:
        bytes_per_param = 2
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    # Memory for model parameters
    model_memory = param_count * bytes_per_param
    
    # Memory for gradients (same size as parameters)
    gradient_memory = model_memory
    
    # Memory for optimizer states (Adam uses 2 states per parameter)
    optimizer_memory = param_count * bytes_per_param * 2
    
    # Effective batch size after gradient accumulation
    effective_batch_size = batch_size // gradient_accumulation_steps
    
    # Estimate activation memory (this is a rough approximation)
    # For transformer models, activations are typically 2-6x the model size depending on sequence length
    # We'll use a conservative estimate based on sequence length
    seq_len = config.max_seq_len
    activation_multiplier = min(6, max(2, seq_len / 512))
    activation_memory = model_memory * activation_multiplier * effective_batch_size
    
    # Memory for attention cache during forward pass
    # Each attention layer stores K and V states of shape [batch_size, seq_len, heads, head_dim]
    head_dim = config.dimension // config.heads
    kv_cache_size_per_layer = 2 * effective_batch_size * seq_len * config.heads * head_dim * bytes_per_param
    kv_cache_memory = kv_cache_size_per_layer * config.layers * config.attention_priority
    
    # Additional memory for MoE routing
    moe_memory = 0
    if config.use_moe:
        # Router logits and indices: [batch_size * seq_len, num_experts]
        router_memory = effective_batch_size * seq_len * config.num_experts * bytes_per_param
        
        # Routing weights: [batch_size * seq_len, num_experts_per_token]
        routing_weights_memory = effective_batch_size * seq_len * config.num_experts_per_token * bytes_per_param
        
        # Expert capacity buffers
        expert_capacity = int(effective_batch_size * seq_len * config.expert_capacity_factor * config.num_experts_per_token / config.num_experts)
        capacity_buffers = config.num_experts * expert_capacity * bytes_per_param
        
        moe_memory = router_memory + routing_weights_memory + capacity_buffers
    
    # Total memory estimate
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory + kv_cache_memory + moe_memory
    
    # Convert to GB
    bytes_to_gb = 1 / (1024 ** 3)
    
    result = {
        'model_parameters_gb': model_memory * bytes_to_gb,
        'gradients_gb': gradient_memory * bytes_to_gb,
        'optimizer_states_gb': optimizer_memory * bytes_to_gb,
        'activations_gb': activation_memory * bytes_to_gb,
        'kv_cache_gb': kv_cache_memory * bytes_to_gb,
        'total_gb': total_memory * bytes_to_gb
    }
    
    if config.use_moe:
        result['moe_routing_gb'] = moe_memory * bytes_to_gb
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate model size from config')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for memory estimation')
    parser.add_argument('--precision', type=str, default='float16', choices=['float32', 'float16', 'bfloat16'], 
                        help='Precision for memory estimation')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()
    
    if args.config:
        config = GPTConfig.from_yaml(args.config)
    else:
        config = GPTConfig.from_yaml(GPTConfig.default_config_path())
    
    # Theoretical calculation
    total_params = calculate_model_size(config)
    
    # PyTorch calculation
    pytorch_params = calculate_pytorch_model_size(config)
    
    print(f"Model Configuration:")
    print(f"Dimension: {config.dimension}")
    print(f"Heads: {config.heads}")
    print(f"Rank: {config.rank}")
    print(f"Layers: {config.layers}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Attention Priority: {config.attention_priority}")
    
    if config.use_moe:
        print(f"Using Mixture of Experts (MoE):")
        print(f"  Number of Experts: {config.num_experts}")
        print(f"  Experts per Token: {config.num_experts_per_token}")
        print(f"  Expert Capacity Factor: {config.expert_capacity_factor}")
    
    print(f"\nTheoretical Parameters: {total_params:,}")
    print(f"PyTorch Parameters: {pytorch_params:,}")
    
    # Break down by component
    embedding_params = config.vocab_size * config.dimension
    
    # Calculate per-layer parameters
    layer_params = (total_params - embedding_params - (config.dimension * config.vocab_size) - (2 * config.dimension if config.bias else config.dimension)) // config.layers
    
    print(f"\nParameter breakdown:")
    print(f"Embeddings: {embedding_params:,}")
    print(f"Per Layer: {layer_params:,}")
    
    if config.use_moe:
        # Router network: dimension -> num_experts
        router_params = config.dimension * config.num_experts
        
        # Each expert has up_proj and down_proj
        expert_params = config.num_experts * (
            # up_proj: dimension -> 4*dimension
            (config.dimension * (4 * config.dimension)) +
            # down_proj: 4*dimension -> dimension
            ((4 * config.dimension) * config.dimension)
        )
        
        # Add bias terms if needed
        if config.bias:
            expert_params += config.num_experts * (
                (4 * config.dimension) +  # up_proj bias
                config.dimension  # down_proj bias
            )
        
        print(f"MoE Router: {router_params:,}")
        print(f"MoE Experts (total): {expert_params:,}")
        print(f"MoE Experts (per expert): {expert_params // config.num_experts:,}")
    
    print(f"Final Layer Norm: {(2 * config.dimension if config.bias else config.dimension):,}")
    print(f"Unembedding: {config.dimension * config.vocab_size + (config.vocab_size if config.bias else 0):,}")
    
    # If there's a discrepancy, print a warning
    if total_params != pytorch_params:
        print(f"\nNote: There is a discrepancy of {abs(total_params - pytorch_params):,} parameters between the theoretical and PyTorch calculations.")
        print("This may be due to additional parameters in the implementation that aren't accounted for in the theoretical calculation.")
    
    # Estimate training memory requirements
    memory_estimate = estimate_training_memory(
        config, 
        batch_size=args.batch_size, 
        precision=args.precision,
        gradient_accumulation_steps=args.grad_accum
    )
    
    print(f"\nEstimated GPU memory requirements (batch_size={args.batch_size}, precision={args.precision}, grad_accum={args.grad_accum}):")
    print(f"Model Parameters: {memory_estimate['model_parameters_gb']:.2f} GB")
    print(f"Gradients: {memory_estimate['gradients_gb']:.2f} GB")
    print(f"Optimizer States: {memory_estimate['optimizer_states_gb']:.2f} GB")
    print(f"Activations: {memory_estimate['activations_gb']:.2f} GB")
    print(f"KV Cache: {memory_estimate['kv_cache_gb']:.2f} GB")
    
    if config.use_moe:
        print(f"MoE Routing: {memory_estimate['moe_routing_gb']:.2f} GB")
    
    print(f"Total Estimated Memory: {memory_estimate['total_gb']:.2f} GB")
    
    # Provide recommendations based on memory requirements
    if memory_estimate['total_gb'] < 8:
        print("\nThis model should fit on most modern GPUs with 8GB+ VRAM.")
    elif memory_estimate['total_gb'] < 16:
        print("\nThis model requires a GPU with at least 16GB VRAM (e.g., RTX 3080, A4000).")
    elif memory_estimate['total_gb'] < 24:
        print("\nThis model requires a high-end GPU with at least 24GB VRAM (e.g., RTX 3090, A5000).")
    elif memory_estimate['total_gb'] < 40:
        print("\nThis model requires a professional GPU with at least 40GB VRAM (e.g., A100, A6000).")
    elif memory_estimate['total_gb'] < 80:
        print("\nThis model requires a high-end data center GPU with at least 80GB VRAM (e.g., A100-80GB).")
    else:
        print("\nThis model is very large and may require model parallelism or multiple GPUs for training.")
    
    # Memory optimization suggestions
    print("\nMemory optimization suggestions:")
    if args.precision == 'float32':
        print("- Use mixed precision training (float16/bfloat16) to reduce memory usage by ~50%")
    
    if args.grad_accum == 1 and memory_estimate['total_gb'] > 16:
        print(f"- Use gradient accumulation (--grad_accum 2 or higher) to reduce memory usage")
    
    if memory_estimate['total_gb'] > 32:
        print("- Consider using DeepSpeed ZeRO or FSDP for distributed training")
    
    if memory_estimate['activations_gb'] > memory_estimate['total_gb'] * 0.3:
        print("- Use gradient checkpointing to trade compute for memory (reduces activation memory)")
    
    if config.use_moe:
        print("- For MoE models, consider:")
        print(f"  - Reducing the number of experts (currently {config.num_experts})")
        print(f"  - Reducing experts per token (currently {config.num_experts_per_token})")
        print("  - Using expert parallelism to distribute experts across multiple GPUs")