#!/bin/bash

# Activate Pipenv environment
source $(pipenv --venv)/bin/activate

# Install dependencies if needed
pipenv install

# Set environment variables for better performance on H100s
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Run distributed training with DeepSpeed
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 8 \
    --multi_gpu \
    src/train_distributed.py \
    --config $1 \
    --deepspeed ds_config.json \
    ${@:2}

# Usage: ./run_h100.sh <config_name> [additional args]
# Example: ./run_h100.sh gpt2_small --resume weights/gpt2_small/checkpoint_latest.pt 