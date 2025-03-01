import argparse
import json
import math
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os

# Import Accelerate for distributed training
from accelerate import Accelerate
from accelerate.utils import set_seed

from models.gpt import GPT
from models.config import GPTConfig
from training.config import TrainingConfig
from training.dataset import ParrotDataset
from training.tokenize import create_tokenizer
from training.metrics import TrainingMetrics

class DistributedTrainer:
    def __init__(self, model_config: GPTConfig, training_config: TrainingConfig, output_dir: Path, accelerator: Accelerate):
        self.config = training_config.trainer
        self.model_config = model_config
        self.accelerator = accelerator
        
        # Initialize model
        self.model = GPT(self.model_config)
        
        # Apply memory optimizations if enabled
        if hasattr(self.config, 'use_gradient_checkpointing') and self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # Log that gradient checkpointing is enabled
            if self.accelerator.is_main_process:
                print("Gradient checkpointing enabled for memory efficiency")
        
        # Initialize tokenizer and dataset
        self.tokenizer = create_tokenizer()
        self.setup_datasets()
        self.setup_optimizer()
        
        # Memory optimization: use bfloat16 mixed precision if available
        if hasattr(self.config, 'memory_efficient') and self.config.memory_efficient:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.mixed_precision = 'bf16'
                if self.accelerator.is_main_process:
                    print("Using bfloat16 mixed precision for memory efficiency")
            else:
                self.mixed_precision = 'fp16'
                if self.accelerator.is_main_process:
                    print("Using fp16 mixed precision for memory efficiency")
        else:
            self.mixed_precision = None
        
        # Prepare model, optimizer, and dataloaders with accelerator
        self.train_dataloader = self.train_dataset.get_dataloader(shuffle=True)
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        # Training state
        self.step = 0
        self.metrics = TrainingMetrics()
        self.tokens_seen = 0
        self.last_log_tokens = 0
        
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
        
        # Create learning rate scheduler
        if self.config.use_lr_schedule:
            self.lr_scheduler = self.get_lr_scheduler()
        else:
            self.lr_scheduler = None
    
    def get_lr_scheduler(self):
        """Create a learning rate scheduler"""
        from transformers import get_scheduler
        
        return get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
    
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
        """Generate a text sample for evaluation"""
        self.model.eval()
        
        # Move prompt to the same device as model
        prompt_tokens = prompt_tokens.to(self.accelerator.device)
        
        # Unwrap model for generation if needed
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        generated = unwrapped_model.generate(
            prompt_tokens,
            max_length=max_length,
            temperature=temperature
        )
        
        self.model.train()
        return self.tokenizer.decode(generated[0].tolist())

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Perform a single training step"""
        input_ids, target_ids = batch
        
        # Forward pass
        logits = self.model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Backward pass and optimization
        self.accelerator.backward(loss)
        
        if (self.step + 1) % self.config.grad_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def train(self):
        """Main training loop"""
        # Load and prepare datasets
        self.train_dataset.load_from_huggingface(
            self.config.dataset.name,
            dataset_config=self.config.dataset.config,
            split=self.config.dataset.train_split,
            text_column=self.config.dataset.text_column
        )
        self.train_dataset.prepare_dataset()
        
        self.eval_dataset.load_from_huggingface(
            self.config.dataset.name,
            dataset_config=self.config.dataset.config,
            split=self.config.dataset.eval_split,
            text_column=self.config.dataset.text_column
        )
        self.eval_dataset.prepare_dataset()
        
        # Training setup
        self.training_start_time = time.time()
        self.last_log_time = self.training_start_time
        
        # Only show progress bar on main process
        if self.accelerator.is_main_process:
            pbar = tqdm(total=self.config.max_steps, initial=self.step)
        
        sample_prompt = "The quick brown fox"
        sample_tokens = torch.tensor([self.tokenizer.encode(sample_prompt)])
        
        # Main training loop
        while self.step < self.config.max_steps:
            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                batch_tokens = batch[0].numel()
                self.tokens_seen += batch_tokens
                
                current_time = time.time()
                tokens_per_sec = (self.tokens_seen - self.last_log_tokens) / (
                    current_time - self.last_log_time
                )
                
                # Gather loss from all processes
                loss = self.accelerator.gather(torch.tensor(loss).to(self.accelerator.device)).mean().item()
                
                # Only log on main process
                if self.accelerator.is_main_process:
                    self.metrics.add_training_step(
                        step=self.step,
                        loss=loss,
                        learning_rate=self.optimizer.param_groups[0]['lr'],
                        tokens_per_sec=tokens_per_sec,
                        perplexity=math.exp(loss)
                    )
                    
                    pbar.update(1)
                    pbar.set_description(
                        f"Loss: {loss:.4f} | PPL: {math.exp(loss):.2f} | "
                        f"Speed: {tokens_per_sec:.0f} tok/s"
                    )
                
                self.step += 1
                
                # Evaluation and checkpointing
                if self.step % self.config.eval_every == 0 and self.accelerator.is_main_process:
                    sample_text = self.generate_sample(sample_tokens)
                    print(f"Sample: {sample_text}")
                    self.metrics.add_generation_sample(self.step, sample_text)
                    self.save_checkpoint()
                
                self.last_log_time = current_time
                self.last_log_tokens = self.tokens_seen
                
                if self.step >= self.config.max_steps:
                    break
        
        # Final checkpoint
        if self.accelerator.is_main_process:
            self.save_checkpoint()
            pbar.close()

    def save_checkpoint(self):
        """Saves model checkpoint and training state"""
        # Create checkpoint directory if it doesn't exist
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Unwrap model before saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        checkpoint = {
            'step': self.step,
            'model_state': unwrapped_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'tokens_seen': self.tokens_seen,
            'metrics': self.metrics.to_dict(),
            'config': self.config,
            'model_config': self.model_config,
        }
        
        # Save latest checkpoint
        latest_path = self.run_dir / 'checkpoint_latest.pt'
        self.accelerator.save(checkpoint, latest_path)
        
        # Save numbered checkpoint every save_every steps
        if self.step % self.config.save_every == 0:
            numbered_path = self.run_dir / f'checkpoint_{self.step:06d}.pt'
            self.accelerator.save(checkpoint, numbered_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Loads model checkpoint and training state"""
        # Load checkpoint to CPU first to avoid OOM issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.step = checkpoint['step']
        self.tokens_seen = checkpoint['tokens_seen']
        self.metrics.from_dict(checkpoint['metrics'])
        
        print(f"Resumed from checkpoint at step {self.step}")

def main():
    parser = argparse.ArgumentParser(description='Train ParrotLM model with distributed training')
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deepspeed', type=str, default="ds_config.json", help='Path to DeepSpeed config')
    parser.add_argument('--memory_efficient', action='store_true', help='Enable memory efficient optimizations')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offloading for optimizer states')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize accelerator with memory optimizations if requested
    accelerator_kwargs = {"log_with": "tensorboard"}
    if args.memory_efficient:
        accelerator_kwargs.update({
            "cpu_offload_model": args.cpu_offload,
            "split_batches": True,
            "dispatch_batches": True
        })
    
    accelerator = Accelerate(**accelerator_kwargs)
    
    # Load training config
    config = TrainingConfig.from_yaml(Path('config') / (args.config + '.yaml'))
    model_config = GPTConfig.from_yaml(args.config)
    
    if args.resume:
        config.resume_from = args.resume
    
    # Add memory optimization flags to config
    config.trainer.use_gradient_checkpointing = args.gradient_checkpointing
    config.trainer.memory_efficient = args.memory_efficient
    
    # Initialize and run trainer
    trainer = DistributedTrainer(
        model_config=model_config,
        training_config=config,
        output_dir=Path('weights') / args.config,
        accelerator=accelerator
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 