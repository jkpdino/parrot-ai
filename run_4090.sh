#!/bin/bash

# Activate Pipenv environment
source $(pipenv --venv)/bin/activate

# Install dependencies if needed
pipenv install

# Set environment variables for better performance on RTX 4090
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run training with mixed precision on single GPU
python src/train.py \
    --config 4090_config \
    ${@:1}

# Usage: ./run_4090.sh [additional args]
# Example: ./run_4090.sh --resume weights/4090_config/checkpoint_latest.pt 