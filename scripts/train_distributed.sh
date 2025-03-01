#!/bin/bash

# This script runs distributed training on multiple GPUs using PyTorch DDP
# Specifically optimized for 2x4090 GPUs

# Default values
CONFIG="moe-nano"
BATCH_SIZE=16
GRAD_ACCUM=1
NUM_GPUS=2
MASTER_PORT=29500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad_accum)
      GRAD_ACCUM="$2"
      shift 2
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create directory for logs
mkdir -p logs

# Set timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_distributed_${CONFIG}_${TIMESTAMP}.log"

echo "Starting distributed training for config: $CONFIG"
echo "Batch size per GPU: $BATCH_SIZE, Gradient accumulation: $GRAD_ACCUM"
echo "Number of GPUs: $NUM_GPUS"
echo "Logs will be saved to: $LOG_FILE"

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Set NCCL environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=lo

# Run the distributed training script
COMMAND="python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  src/train_distributed.py \
  --config $CONFIG \
  --batch_size $BATCH_SIZE \
  --gradient_checkpointing \
  --mixed_precision \
  --monitor_memory"

# Add resume flag if provided
if [ ! -z "$RESUME" ]; then
  COMMAND="$COMMAND --resume $RESUME"
fi

# Run the command and log output
echo "Running command: $COMMAND"
$COMMAND 2>&1 | tee "$LOG_FILE"

echo "Training completed. Log saved to: $LOG_FILE" 