#!/bin/bash

# Activate Pipenv environment
source $(pipenv --venv)/bin/activate

# Install dependencies if needed
pipenv install

# Set environment variables for better performance and memory efficiency on RTX 4090s
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_P2P_LEVEL=NVL

# Memory optimization settings
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Run distributed training with DeepSpeed on 2 GPUs
accelerate launch \
    --config_file accelerate_config_dual_4090.yaml \
    --num_processes 2 \
    --multi_gpu \
    src/train_distributed.py \
    --config dual_4090 \
    --deepspeed ds_config_dual_4090.json \
    --memory_efficient \
    --gradient_checkpointing \
    ${@:1}

# Usage: ./run_dual_4090.sh [additional args]
# Example: ./run_dual_4090.sh --resume weights/dual_4090/checkpoint_latest.pt 