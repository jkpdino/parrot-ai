import argparse
import json
import math  # Add this import
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional, Tuple

from models.gpt import GPT
from models.config import GPTConfig
from training.config import TrainingConfig
from training.dataset import ParrotDataset
from training.tokenize import create_tokenizer
from training.metrics import TrainingMetrics

class Trainer:
    def __init__(self, model_config: GPTConfig, training_config: TrainingConfig, output_dir: Path):
        self.config = training_config.trainer
        self.model_config = model_config
        
        # Initialize model and optimizer
        self.model = GPT(self.model_config)
        self.setup_optimizer()
        
        # Initialize tokenizer and dataset
        self.tokenizer = create_tokenizer()
        self.setup_datasets()
        
        # Training state
        self.step = 0
        self.metrics = TrainingMetrics()

        self.run_dir = output_dir
        
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
    
    def setup_optimizer(self):
        """Initialize optimizer based on config"""
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                learning_rate=float(self.config.learning_rate),
                weight_decay=float(self.config.weight_decay)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def setup_datasets(self):
        """Initialize training and evaluation datasets"""
        self.train_dataset = ParrotDataset(
            self.tokenizer,
            max_length=self.config.max_length,
            batch_size=self.config.train_batch_size
        )
        
        self.eval_dataset = ParrotDataset(
            self.tokenizer,
            max_length=self.config.max_length,
            batch_size=self.config.eval_batch_size
        )
    
    def save_checkpoint(self):
        """Save model checkpoint and training state"""
        checkpoint_path = self.run_dir / f"checkpoint_{self.step}.safetensors"
        self.model.save_weights(checkpoint_path.as_posix())
        
        # Save metrics separately for easier analysis
        self.metrics.save(self.run_dir)
        
        # Save latest checkpoint reference
        with open(self.run_dir / "latest.txt", 'w') as f:
            f.write(str(checkpoint_path))
    
    def generate_sample(self, prompt_tokens: mx.array, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text from a prompt during training.
        
        Args:
            prompt_tokens: Starting token sequence
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
        """
        generated = self.model.generate(
            prompt_tokens,
            max_length=max_length,
            temperature=temperature
        )
        return self.tokenizer.decode(generated[0].tolist())

    def evaluate_generation(self, num_samples: int = 5) -> Dict[str, float]:
        """Enhanced evaluation with multiple metrics"""
        metrics = {
            'loss': 0.0,
            'perplexity': 0.0,
            'tokens_per_sec': 0.0
        }
        
        start_time = time.time()
        total_tokens = 0
        
        eval_batch = next(self.eval_dataset.get_batch_iterator(shuffle=True))
        input_ids, target_ids = eval_batch
        
        for i in range(min(num_samples, len(input_ids))):
            seq_len = len(input_ids[i])
            prompt_len = seq_len // 2
            prompt = input_ids[i:i+1, :prompt_len]
            
            generated = self.model.generate(
                prompt,
                max_length=seq_len,
                temperature=0.8
            )
            
            target = target_ids[i:i+1, prompt_len:]
            pred = generated[:, prompt_len:]
            loss = mx.mean(nn.losses.cross_entropy(pred, target))
            
            metrics['loss'] += loss.item()
            total_tokens += len(target.reshape(-1))
        
        # Average metrics
        metrics['loss'] /= num_samples
        metrics['perplexity'] = self.calculate_perplexity(metrics['loss'])
        metrics['tokens_per_sec'] = total_tokens / (time.time() - start_time)
        
        return metrics

    def train(self):
        """Main training loop with enhanced monitoring"""
        # Setup datasets
        self.train_dataset.load_from_huggingface(
            self.config.dataset_name,
            split=self.config.dataset_split,
            text_column=self.config.text_column
        )
        self.train_dataset.prepare_dataset()
        
        self.eval_dataset.load_from_huggingface(
            self.config.dataset_name,
            split=self.config.eval_split,
            text_column=self.config.text_column
        )
        self.eval_dataset.prepare_dataset()
        
        pbar = tqdm(total=self.config.max_steps, initial=self.step)
        
        # Add sample generation prompt
        sample_prompt = "The quick brown fox"
        sample_tokens = mx.array([self.tokenizer.encode(sample_prompt)])
        
        self.training_start_time = time.time()
        self.last_log_time = self.training_start_time
        pbar = tqdm(total=self.config.max_steps, initial=self.step)
        
        while self.step < self.config.max_steps:
            for batch in self.train_dataset.get_batch_iterator():
                # Training step
                loss, batch_tokens = self.train_step(batch)
                self.tokens_seen += batch_tokens
                
                # Calculate training speed
                current_time = time.time()
                tokens_per_sec = (self.tokens_seen - self.last_log_tokens) / (
                    current_time - self.last_log_time
                )
                
                # Log metrics
                self.metrics.add_training_step(
                    step=self.step,
                    loss=loss,
                    learning_rate=self.optimizer.learning_rate.item(),
                    tokens_per_sec=tokens_per_sec,
                    perplexity=self.calculate_perplexity(loss)
                )
                
                # Update progress
                self.step += 1
                pbar.update(1)
                pbar.set_description(
                    f"Loss: {loss:.4f} | PPL: {self.calculate_perplexity(loss):.2f} | "
                    f"Speed: {tokens_per_sec:.0f} tok/s"
                )
                
                # Periodic evaluation
                if self.step % self.config.eval_every == 0:
                    eval_metrics = self.evaluate_generation()
                    self.metrics.add_eval_step(**eval_metrics)
                    
                    sample_text = self.generate_sample(sample_tokens)
                    self.metrics.add_generation_sample(self.step, sample_text)
                    
                    self.save_checkpoint()
                
                # Update monitoring state
                self.last_log_time = current_time
                self.last_log_tokens = self.tokens_seen
                
                if self.step >= self.config.max_steps:
                    break
        
        # Final checkpoint
        self.save_checkpoint()
        pbar.close()

    def generate(self, 
                text: str, 
                max_length: int = 100, 
                temperature: float = 0.8,
                use_tqdm: bool = True) -> str:
        """Generate text from a prompt (for inference).
        
        Args:
            text: Input prompt text
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            use_tqdm: Whether to show progress bar
        """
        tokens = mx.array([self.tokenizer.encode(text)])
        
        with mx.stop_gradient():
            generated = self.model.generate(
                tokens,
                max_length=max_length,
                temperature=temperature
            )
            
        return self.tokenizer.decode(generated[0].tolist())

    def train_step(self, batch: Tuple[mx.array, mx.array]) -> float:
        """Perform a single training step.
        
        Args:
            batch: Tuple of (input_ids, target_ids)
            
        Returns:
            float: Training loss for this step
        """
        input_ids, target_ids = batch
        
        def loss_fn(model):
            logits = model(input_ids)
            return mx.mean(nn.losses.cross_entropy(
                logits, input_ids
            ))
        
        # Get loss and gradients
        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        
        # Accumulate gradients
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            self.accumulated_grads = tree_map(
                lambda g1, g2: g1 + g2,
                self.accumulated_grads,
                grads
            )
        self.accumulation_count += 1
        
        # Update parameters when we've accumulated enough steps
        if self.accumulation_count >= self.config.grad_accumulation_steps:
            # Scale gradients by accumulation steps
            scaled_grads = tree_map(
                lambda g: g * (1.0 / self.config.grad_accumulation_steps),
                self.accumulated_grads
            )
            
            # Update parameters
            self.optimizer.update(self.model, scaled_grads)
            
            # Reset accumulation
            self.accumulated_grads = None
            self.accumulation_count = 0
            
            # Update learning rate schedule
            if self.config.use_lr_schedule:
                self._update_learning_rate()
        
        # Ensure computations are done
        mx.eval(loss, self.model.parameters())
        
        return loss.item()

    def _update_learning_rate(self):
        """Update learning rate according to schedule"""
        if self.step < self.config.warmup_steps:
            # Linear warmup
            lr = self.config.learning_rate * (self.step / self.config.warmup_steps)
        else:
            # Cosine decay with minimum learning rate
            progress = (self.step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            lr = self.config.min_lr + 0.5 * (self.config.learning_rate - self.config.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        
        self.optimizer.learning_rate = mx.array(lr)

def main():
    parser = argparse.ArgumentParser(description='Train ParrotLM model')
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load training config
    config = TrainingConfig.from_yaml(Path('config') / (args.config + '.yaml'))
    model_config = GPTConfig.from_yaml(args.config)
    if args.resume:
        config.resume_from = args.resume
    
    # Initialize and run trainer
    trainer = Trainer(model_config, config, Path('weights') / args.config)
    trainer.train()

if __name__ == "__main__":
    main()