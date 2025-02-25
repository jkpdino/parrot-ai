from models.config import GPTConfig
import argparse
from models.gpt import GPT

import mlx

def calculate_model_size(config: GPTConfig):
    # Embedding layers
    embedding_params = config.vocab_size * config.dimension  # Token embeddings
    
    # Low Rank Attention parameters (per attention block)
    attention_params = (
        # Q, K, V projections (up and down)
        (config.dimension * config.rank + config.rank * config.dimension) * 3
    )
    
    # MLP parameters
    mlp_params = (
        # First linear layer
        (config.dimension * (4 * config.dimension)) +
        # Second linear layer
        ((4 * config.dimension) * config.dimension)
    )
    if config.bias:
        mlp_params += (4 * config.dimension) + config.dimension  # Bias terms
    
    # Layer norm parameters (if using bias)
    layer_norm_params = 2 * config.dimension if config.bias else config.dimension
    
    # Total parameters per transformer block
    block_params = (
        (attention_params + layer_norm_params) * config.attention_priority +  # Attention blocks
        mlp_params + layer_norm_params  # MLP block
    )
    
    # Final unembedding layer
    unembed_params = config.dimension * config.vocab_size
    if config.bias:
        unembed_params += config.vocab_size
    
    # Total parameters
    total_params = (
        embedding_params +
        (block_params * config.layers) +
        unembed_params
    )
    
    return total_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate model size from config')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    args = parser.parse_args()
    
    if args.config:
        config = GPTConfig.from_yaml(args.config)
    else:
        config = GPTConfig.from_yaml(GPTConfig.default_config_path())
    
    # Theoretical calculation
    total_params = calculate_model_size(config)
    
    print(f"Model Configuration:")
    print(f"Dimension: {config.dimension}")
    print(f"Heads: {config.heads}")
    print(f"Rank: {config.rank}")
    print(f"Layers: {config.layers}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Attention Priority: {config.attention_priority}")
    print(f"\nTheoretical Parameters: {total_params:,}")
    
    # Break down by component
    embedding_params = config.vocab_size * config.dimension
    print(f"\nParameter breakdown:")
    print(f"Embeddings: {embedding_params:,}")
    print(f"Per Layer: {(total_params - embedding_params - (config.dimension * config.vocab_size)) // config.layers:,}")
    print(f"Unembedding: {config.dimension * config.vocab_size:,}")