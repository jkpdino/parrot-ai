#!/bin/bash

# This script runs the ParrotLM training with memory optimizations
# to reduce GPU memory usage.

# Default values
CONFIG="moe-nano"
BATCH_SIZE=8
GRAD_ACCUM=2

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
LOG_FILE="logs/train_${CONFIG}_${TIMESTAMP}.log"

echo "Starting memory-optimized training for config: $CONFIG"
echo "Batch size: $BATCH_SIZE, Gradient accumulation: $GRAD_ACCUM"
echo "Logs will be saved to: $LOG_FILE"

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Run the training script with memory optimizations
COMMAND="python src/train.py \
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