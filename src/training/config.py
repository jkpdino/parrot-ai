from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional, Dict, Any

@dataclass
class TrainerConfig:
    # Optimization
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    gradient_clip: float = 1.0
    min_lr: float = 1e-5  # Added for cosine decay
    use_lr_schedule: bool = True  # Added for LR scheduling
    grad_accumulation_steps: int = 1  # Added for gradient accumulation
    
    # Training loop
    max_steps: int = 100000
    save_every: int = 1000
    eval_every: int = 500
    eval_batches: int = 100
    log_every: int = 10
    
    # Monitoring
    num_eval_samples: int = 5  # Added for evaluation
    eval_max_tokens: int = 200  # Added for evaluation
    eval_temperature: float = 0.8  # Added for evaluation
    
    # Dataset
    dataset_name: str = "wikitext"
    dataset_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_length: int = 512
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    resume_from: Optional[str] = None

@dataclass
class TrainingConfig:
    model: Dict[str, Any]
    trainer: TrainerConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        trainer_config = TrainerConfig(**config_dict.get('training', {}))
        return cls(
            model=config_dict.get('model', {}),
            trainer=trainer_config
        )