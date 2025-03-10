import argparse
import json
import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os

from models.gpt import GPT
from models.config import GPTConfig
from training.config import TrainingConfig
from training.dataset import ParrotDataset
from training.tokenize import create_tokenizer
from training.metrics import TrainingMetrics
from utils.memory_monitor import MemoryMonitor

class Trainer:
    def __init__(self, model_config: GPTConfig, training_config: TrainingConfig, output_dir: Path):
        self.config = training_config.trainer
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and move to device
        self.model = GPT(self.model_config).to(self.device)
        
        # Apply memory optimizations if enabled
        if hasattr(self.config, 'use_gradient_checkpointing') and self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for memory efficiency")
            
        self.setup_optimizer()
        
        # Initialize tokenizer and dataset
        self.tokenizer = create_tokenizer()
        self.setup_datasets()
        
        # Training state
        self.step = 0
        self.metrics = TrainingMetrics()
        self.scaler = GradScaler()  # For mixed precision training
        self.accumulated_grads = None
        self.accumulation_count = 0
        self.tokens_seen = 0
        self.last_log_tokens = 0
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(log_interval=30, enabled=True)
        
        self.run_dir = output_dir
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
    
    def setup_optimizer(self):
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.learning_rate),
                weight_decay=float(self.config.weight_decay)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def setup_datasets(self):
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
    
    @torch.no_grad()
    def generate_sample(self, prompt_tokens: torch.Tensor, max_length: int = 100, temperature: float = 0.8) -> str:
        self.model.eval()
        generated = self.model.generate(
            prompt_tokens.to(self.device),
            max_length=max_length,
            temperature=temperature
        )
        self.model.train()
        return self.tokenizer.decode(generated[0])

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        input_ids, target_ids = [x.to(self.device) for x in batch]
        
        # Mixed precision training with device awareness
        with autocast(enabled=hasattr(self.config, 'use_mixed_precision') and self.config.use_mixed_precision):
            logits = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        
        # Scale loss and backward pass
        scaled_loss = loss / self.config.grad_accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        
        # Update parameters when we've accumulated enough steps
        if self.accumulation_count >= self.config.grad_accumulation_steps - 1:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Step optimizer with scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
            self.accumulation_count = 0
            
            if self.config.use_lr_schedule:
                self._update_learning_rate()
        else:
            self.accumulation_count += 1
        
        return loss.item()

    def train(self):
        self.train_dataset.load_from_huggingface(
            self.config.dataset.name,
            dataset_config=self.config.dataset.config,
            split=self.config.dataset.train_split,
            text_column=self.config.dataset.text_column
        )
        self.train_dataset.prepare_dataset()
        train_loader = self.train_dataset.get_dataloader(shuffle=True)

        self.eval_dataset.load_from_huggingface(
            self.config.dataset.name,
            dataset_config=self.config.dataset.config,
            split=self.config.dataset.eval_split,
            text_column=self.config.dataset.text_column
        )
        self.eval_dataset.prepare_dataset()
        
        self.training_start_time = time.time()
        self.last_log_time = self.training_start_time
        pbar = tqdm(total=self.config.max_steps, initial=self.step)
        
        sample_prompt = "The quick brown fox"
        sample_tokens = torch.tensor([self.tokenizer.encode(sample_prompt)])
        
        # Start memory monitoring
        self.memory_monitor.start()
        
        try:
            while self.step < self.config.max_steps:
                for batch in train_loader:
                    loss = self.train_step(batch)
                    batch_tokens = batch[0].numel()
                    self.tokens_seen += batch_tokens
                    
                    current_time = time.time()
                    tokens_per_sec = (self.tokens_seen - self.last_log_tokens) / (
                        current_time - self.last_log_time
                    )
                    
                    self.metrics.add_training_step(
                        step=self.step,
                        loss=loss,
                        learning_rate=self.optimizer.param_groups[0]['lr'],
                        tokens_per_sec=tokens_per_sec,
                        perplexity=math.exp(loss)
                    )
                    
                    self.step += 1
                    pbar.update(1)
                    pbar.set_description(
                        f"Loss: {loss:.4f} | PPL: {math.exp(loss):.2f} | "
                        f"Speed: {tokens_per_sec:.0f} tok/s"
                    )
                    
                    if self.step % self.config.eval_every == 0:
                        # Clear cache before evaluation to free up memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        sample_text = self.generate_sample(sample_tokens)
                        print(f"Sample: {sample_text}")
                        self.metrics.add_generation_sample(self.step, sample_text)
                        self.save_checkpoint()
                        
                        # Log memory usage after evaluation
                        self.memory_monitor.log_memory_usage()
                    
                    self.last_log_time = current_time
                    self.last_log_tokens = self.tokens_seen
                    
                    if self.step >= self.config.max_steps:
                        break
            
            self.save_checkpoint()
        finally:
            # Stop memory monitoring
            self.memory_monitor.stop()
            
            # Log peak memory usage
            peak_memory = self.memory_monitor.get_peak_memory()
            print(f"Peak GPU memory: {peak_memory.get('peak_gpu_gb', 0):.2f} GB")
            print(f"Peak CPU memory: {peak_memory.get('peak_cpu_gb', 0):.2f} GB")
            
            # Check for potential memory leaks
            leaks = self.memory_monitor.detect_leaks()
            if leaks:
                print("Potential memory leaks detected:")
                for leak in leaks:
                    print(f"  - {leak}")
            
            pbar.close()

    @torch.no_grad()
    def evaluate_generation(self, num_samples: int = 5) -> Dict[str, float]:
        """Enhanced evaluation with multiple metrics"""
        metrics = {
            'loss': 0.0,
            'perplexity': 0.0,
            'tokens_per_sec': 0.0
        }
        
        start_time = time.time()
        total_tokens = 0

        if not hasattr(self.eval_dataset, 'dataset') or len(self.eval_dataset) == 0:
            self.eval_dataset.load_from_huggingface(
                self.config.dataset.name,
                dataset_config=self.config.dataset.config,
                split=self.config.dataset.eval_split,
                text_column=self.config.dataset.text_column
            )
            self.eval_dataset.prepare_dataset()
        train_loader = self.eval_dataset.get_dataloader(shuffle=True)


        for i, batch in enumerate(train_loader):
            if i >= num_samples:
                break
            input_ids, target_ids = [x.to(self.device) for x in batch]
                    
            logits = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )

            metrics['loss'] += loss.item()
            total_tokens += input_ids.numel()
        
        # Average metrics
        metrics['loss'] /= num_samples
        metrics['perplexity'] = math.exp(metrics['loss'])
        metrics['tokens_per_sec'] = total_tokens / (time.time() - start_time)
        
        return metrics

    @torch.no_grad
    def generate(self, 
                text: str, 
                max_length: int = 100, 
                temperature: float = 0.8,
                use_tqdm: bool = True) -> str:
        """Generate text from a prompt (for inference)."""
        tokens = torch.tensor([self.tokenizer.encode(text)]).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                tokens,
                max_length=max_length,
                temperature=temperature
            )
            
        return self.tokenizer.decode(generated[0].tolist())

    def _update_learning_rate(self):
        """Update learning rate according to schedule"""
        if self.step < self.config.warmup_steps:
            # Linear warmup
            lr = float(self.config.learning_rate) * (self.step / self.config.warmup_steps)
        else:
            # Cosine decay with minimum learning rate
            progress = (self.step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            lr = float(self.config.min_lr) + 0.5 * (float(self.config.learning_rate) - float(self.config.min_lr)) * (
                1.0 + math.cos(math.pi * progress)
            )
        
        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self):
        """Saves model checkpoint and training state"""
        checkpoint = {
            'step': self.step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'tokens_seen': self.tokens_seen,
            'metrics': self.metrics.to_dict(),
            'config': self.config,
            'model_config': self.model_config,
        }
        
        # Create checkpoint directory if it doesn't exist
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save latest checkpoint
        latest_path = self.run_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save numbered checkpoint every save_every steps
        if self.step % self.config.save_every == 0:
            numbered_path = self.run_dir / f'checkpoint_{self.step:06d}.pt'
            torch.save(checkpoint, numbered_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Loads model checkpoint and training state"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        self.step = checkpoint['step']
        self.tokens_seen = checkpoint['tokens_seen']
        self.metrics.from_dict(checkpoint['metrics'])
        
        print(f"Resumed from checkpoint at step {self.step}")

def main():
    parser = argparse.ArgumentParser(description='Train ParrotLM model')
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing for memory efficiency')
    parser.add_argument('--batch_size', type=int, help='Override batch size from config')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--monitor_memory', action='store_true', help='Enable memory monitoring')
    
    args = parser.parse_args()
    
    # Load training config
    config = TrainingConfig.from_yaml(Path('config') / (args.config + '.yaml'))
    model_config = GPTConfig.from_yaml(args.config)
    if args.resume:
        config.resume_from = args.resume
    
    # Apply memory optimization flags
    config.trainer.use_gradient_checkpointing = args.gradient_checkpointing
    config.trainer.use_mixed_precision = args.mixed_precision
    
    if args.mixed_precision:
        print("Mixed precision training enabled for memory efficiency")
    
    # Override batch size if specified
    if args.batch_size:
        config.trainer.dataset.train_batch_size = args.batch_size
        print(f"Overriding batch size to {args.batch_size}")
    
    # Initialize and run trainer
    trainer = Trainer(model_config, config, Path('weights') / args.config)
    trainer.train()

if __name__ == "__main__":
    main()