# Memory Optimization Guide for ParrotLM

This guide explains how to optimize memory usage when training ParrotLM models, especially for Mixture of Experts (MoE) architectures.

## Understanding Memory Usage

When training MoE models, memory usage can be significantly higher than standard transformer models due to:

1. **Expert Parameters**: Each expert adds parameters that need to be stored in memory
2. **Routing Mechanism**: The token routing process creates additional temporary tensors
3. **Capacity Control**: The capacity control mechanism requires additional buffers
4. **Activation Memory**: MoE models can have higher activation memory due to the parallel processing of experts

## Memory Optimization Techniques

We've implemented several memory optimization techniques:

### 1. Gradient Checkpointing

Gradient checkpointing trades computation for memory by recomputing activations during the backward pass instead of storing them. This can reduce memory usage by 30-50% with a small performance penalty.

```bash
python src/train.py --config moe-nano --gradient_checkpointing
```

### 2. Mixed Precision Training

Using FP16 or BF16 precision can reduce memory usage by almost 50% compared to FP32.

```bash
python src/train.py --config moe-nano --mixed_precision
```

### 3. Batch Size and Gradient Accumulation

Reducing batch size is the most direct way to reduce memory usage. You can maintain effective batch size using gradient accumulation.

```bash
python src/train.py --config moe-nano --batch_size 8
```

### 4. Memory-Efficient Implementation

We've optimized the MoE implementation to:

- Use in-place operations where possible
- Reuse tensors instead of creating new ones
- Implement a more memory-efficient routing mechanism
- Avoid unnecessary tensor allocations

### 5. Memory Monitoring

We've added a memory monitoring utility to track memory usage during training:

```bash
python src/train.py --config moe-nano --monitor_memory
```

## Using the Memory-Optimized Training Script

For convenience, we've created a script that applies all memory optimizations:

```bash
./scripts/train_memory_optimized.sh --config moe-nano --batch_size 8
```

Options:

- `--config`: Model configuration (default: moe-nano)
- `--batch_size`: Training batch size (default: 8)
- `--grad_accum`: Gradient accumulation steps (default: 2)
- `--resume`: Path to checkpoint to resume from

## Recommended Settings for Different GPU Sizes

| GPU Memory | Batch Size | Gradient Accumulation | Other Optimizations                      |
| ---------- | ---------- | --------------------- | ---------------------------------------- |
| 8GB        | 4          | 4                     | All                                      |
| 12GB       | 8          | 2                     | All                                      |
| 16GB       | 16         | 1                     | Mixed Precision + Gradient Checkpointing |
| 24GB+      | 32         | 1                     | Mixed Precision                          |

## Troubleshooting Memory Issues

If you're still experiencing memory issues:

1. **Reduce Model Size**:

   - Reduce the number of experts (currently 4)
   - Reduce the number of experts per token (currently 2)
   - Reduce the expert capacity factor (currently 1.25)

2. **Enable PyTorch Memory Profiling**:

   ```python
   from torch.profiler import profile, record_function, ProfilerActivity

   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       # Your training code here
   print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
   ```

3. **Clear Cache Regularly**:

   ```python
   torch.cuda.empty_cache()
   ```

4. **Use Distributed Training**:
   For very large models, consider using the distributed training script with DeepSpeed or FSDP:
   ```bash
   python src/train_distributed.py --config moe-nano
   ```

## Memory Usage Comparison

| Configuration | Original Memory Usage | Optimized Memory Usage |
| ------------- | --------------------- | ---------------------- |
| moe-nano      | ~24GB                 | ~4-6GB                 |

## Additional Resources

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
