import yaml
from pathlib import Path

class GPTConfig:
    dimension: int = 1024
    heads: int = 16
    rank: int = 32

    layers: int = 2
    max_seq_len: int = 1024
    vocab_size: int = 50257

    attention_priority: int = 2

    dropout: float = 0.0
    bias: bool = True
    
    # Mixture of Experts parameters
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.0
    moe_jitter_eps: float = 0.0
    moe_dropout: float = 0.0

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(str(Path('config') / (yaml_path + '.yaml')), 'r') as f:
            config = yaml.safe_load(f)
        
        instance = cls()
        model_config = config.get('model', {})
        for key, value in model_config.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    @classmethod
    def default_config_path(cls):
        return 'base'