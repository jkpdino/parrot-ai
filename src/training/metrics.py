from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
from pathlib import Path
import time
from datetime import datetime

@dataclass
class TrainingMetrics:
    # Existing fields
    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    best_eval_loss: float = float('inf')
    
    # New monitoring fields
    train_perplexities: List[float] = field(default_factory=list)
    eval_perplexities: List[float] = field(default_factory=list)
    tokens_per_second: List[float] = field(default_factory=list)
    total_tokens: List[int] = field(default_factory=list)
    generation_metrics: List[Dict[str, float]] = field(default_factory=list)
    generation_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_training_step(
        self, 
        step: int, 
        loss: float, 
        learning_rate: float,
        tokens_per_sec: float,
        perplexity: float
    ):
        """Add training metrics for a step."""
        self.train_losses.append(loss)
        self.learning_rates.append(learning_rate)
        self.steps.append(step)
        self.timestamps.append(time.time())
        self.tokens_per_second.append(tokens_per_sec)
        self.train_perplexities.append(perplexity)
    
    def add_eval_step(self, **metrics):
        """Add evaluation metrics.
        
        Args:
            metrics: Dictionary containing evaluation metrics
                - loss: float
                - perplexity: float
                - tokens_per_sec: float
        """
        self.eval_losses.append(metrics['loss'])
        self.eval_perplexities.append(metrics['perplexity'])
        self.generation_metrics.append(metrics)
        
        if metrics['loss'] < self.best_eval_loss:
            self.best_eval_loss = metrics['loss']

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