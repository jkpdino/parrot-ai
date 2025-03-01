from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional, Dict, Any, List

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    name: str = "wikitext"
    config: Optional[str] = "wikitext-2-v1"
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_length: int = 512

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
    grad_accumulation_steps: int = 8  # Added for gradient accumulation
    
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
    
    # Dataset - for backward compatibility
    dataset_name: str = "wikitext"
    dataset_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_length: int = 512
    
    # New dataset configuration
    dataset: Optional[DatasetConfig] = None
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    resume_from: Optional[str] = None
    
    def __post_init__(self):
        """Initialize dataset config if not provided but using new config format."""
        if self.dataset is None:
            # Check if we're using the new format with dataset field
            # If not, create a dataset config from the flat fields for backward compatibility
            self.dataset = DatasetConfig(
                name=self.dataset_name,
                config=None,  # Default to None for backward compatibility
                train_split=self.dataset_split,
                eval_split=self.eval_split,
                text_column=self.text_column,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
                max_length=self.max_length
            )

@dataclass
class TrainingConfig:
    model: Dict[str, Any]
    trainer: TrainerConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        training_dict = config_dict.get('training', {})
        
        # Handle nested dataset configuration if present
        dataset_dict = training_dict.pop('dataset', None)
        trainer_config = TrainerConfig(**training_dict)
        
        if dataset_dict:
            trainer_config.dataset = DatasetConfig(**dataset_dict)
        
        return cls(
            model=config_dict.get('model', {}),
            trainer=trainer_config
        )