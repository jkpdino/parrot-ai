from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
from pathlib import Path
import time
import torch
from torch.utils.tensorboard import SummaryWriter

@dataclass
class TrainingMetrics:
    writer: SummaryWriter = None
    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    best_eval_loss: float = float('inf')
    
    train_perplexities: List[float] = field(default_factory=list)
    eval_perplexities: List[float] = field(default_factory=list)
    tokens_per_second: List[float] = field(default_factory=list)
    generation_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.writer is None:
            self.writer = SummaryWriter()
    
    def add_training_step(
        self, 
        step: int, 
        loss: float, 
        learning_rate: float,
        tokens_per_sec: float,
        perplexity: float
    ):
        self.train_losses.append(loss)
        self.learning_rates.append(learning_rate)
        self.steps.append(step)
        self.timestamps.append(time.time())
        self.tokens_per_second.append(tokens_per_sec)
        self.train_perplexities.append(perplexity)
        
        # Log to tensorboard
        self.writer.add_scalar('train/loss', loss, step)
        self.writer.add_scalar('train/perplexity', perplexity, step)
        self.writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
        self.writer.add_scalar('train/learning_rate', learning_rate, step)

    def add_eval_step(self, **metrics):
        self.eval_losses.append(metrics['loss'])
        self.eval_perplexities.append(metrics['perplexity'])
        
        if metrics['loss'] < self.best_eval_loss:
            self.best_eval_loss = metrics['loss']
            
        # Log to tensorboard
        step = self.steps[-1] if self.steps else 0
        self.writer.add_scalar('eval/loss', metrics['loss'], step)
        self.writer.add_scalar('eval/perplexity', metrics['perplexity'], step)

    def add_generation_sample(self, step: int, text: str):
        """Add a generated text sample with metadata."""
        self.generation_samples.append({
            'step': step,
            'text': text,
            'timestamp': time.time()
        })

    def save(self, run_dir: Path):
        metrics_data = {
            # Existing metrics
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
            'steps': self.steps,
            'timestamps': self.timestamps,
            'best_eval_loss': self.best_eval_loss,
            
            # New metrics
            'train_perplexities': self.train_perplexities,
            'eval_perplexities': self.eval_perplexities,
            'tokens_per_second': self.tokens_per_second,
            'generation_metrics': self.generation_metrics,
            'generation_samples': self.generation_samples
        }
        
        with open(run_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load(self, path: Path):
        with open(path / 'metrics.json', 'r') as f:
            metrics_dict = json.load(f)
        
        # Load all fields
        for key, value in metrics_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)