# Checkpoint & Resume Behavior

## Overview

GDSearch provides robust checkpointing and resume functionality for long-running experiments. This document explains what is saved, what is not, and best practices for reproducible experiments.

## RNG State Handling

### ✅ What IS Saved

The checkpoint system saves the following RNG states:
- **Python random:** `random.getstate()`
- **NumPy random:** `np.random.get_state()`
- **PyTorch CPU:** `torch.get_rng_state()`
- **PyTorch CUDA:** `torch.cuda.get_rng_state_all()` (all GPUs)

### ❌ What is NOT Saved

**DataLoader worker RNG states** are NOT captured by checkpoints:
- Each DataLoader worker process has independent RNG state
- Worker iteration position (which batch was last processed) is not tracked
- PyTorch DataLoader does not expose worker states via public API

## Known Limitation: Mid-Epoch Resume

If you resume from a checkpoint saved **mid-epoch**, training will restart from the **beginning of that epoch** with potentially different batch order.

**Why this happens:**
1. DataLoader workers re-initialize with base seed
2. Shuffle order may differ from original training run
3. Iteration position within epoch is not preserved

**Impact on reproducibility:**
- ✅ **Epoch-boundary resumes:** Fully reproducible
- ⚠️ **Mid-epoch resumes:** Loss trajectory may differ slightly due to different batch order

## Best Practices for Reproducibility

### 1. Save Checkpoints at Epoch Boundaries Only

```python
# RECOMMENDED: Save after epoch completes
if epoch % checkpoint_interval == 0:
    checkpoint_manager.save_checkpoint(...)

# AVOID: Saving mid-epoch (e.g., every N batches)
if batch_idx % save_every_n_batches == 0:
    checkpoint_manager.save_checkpoint(...)  # ❌ Not fully reproducible
```

### 2. Use Deterministic Training for Critical Experiments

```bash
# Enable full determinism
python run_all_kaggle.py --deterministic --seeds 42,123,456

# This sets:
# - torch.use_deterministic_algorithms(True)
# - CUBLAS_WORKSPACE_CONFIG environment variable
# - Disables non-deterministic operations
```

### 3. Verify Resume Equivalence

Use the built-in resume verification test:

```bash
# Golden test: Train(10) == Train(5)->Save->Load->Train(5)
python run_all_kaggle.py --verify-resume --experiments mnist
```

This ensures that:
1. Training from scratch for 10 epochs
2. Produces same results as training 5 epochs, checkpointing, resuming for 5 more

## Resume Behavior Modes

You can control what happens when resuming without a checkpoint:

```bash
# Error if no checkpoint found (fail-fast for debugging)
python run_all_kaggle.py --resume --resume-behavior error_if_no_checkpoint

# Restart from scratch if no checkpoint (default with --resume)
python run_all_kaggle.py --resume --resume-behavior skip_if_results_exist

# Always restart if no checkpoint (ignore existing results)
python run_all_kaggle.py --resume --resume-behavior restart_if_no_checkpoint
```

## What Gets Checkpointed

### Model State
- All model parameters (weights and biases)
- Batch normalization running statistics
- Dropout states (training vs eval mode)

### Optimizer State
- Momentum buffers (SGD with momentum)
- First and second moment estimates (Adam/AdamW)
- Maximum second moment (AMSGrad)
- Slow weights (Lookahead)
- Per-parameter adaptive learning rates (AdaBound, AdaBelief)

### Scheduler State
- Current learning rate
- Step count and cycle position (CosineAnnealingLR)
- Best metric tracker (ReduceLROnPlateau)

### Training Progress
- Current epoch number
- Training history (loss, accuracy per epoch)
- Best validation metrics
- Time elapsed

### Global RNG States
- Python, NumPy, PyTorch CPU/CUDA random generators

## Advanced: Custom Checkpoint Saving

For custom training loops, use the CheckpointManager:

```python
from src.core.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir="artifacts/checkpoints",
    max_checkpoints=5,  # Keep only 5 most recent
    save_interval=5     # Save every 5 epochs
)

# Save checkpoint
checkpoint_data = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'train_loss': train_loss,
    'val_accuracy': val_accuracy,
    'rng_state': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
}

checkpoint_manager.save_checkpoint(
    checkpoint_data,
    filename=f"model_epoch{epoch}.pt",
    experiment_name="my_experiment"
)

# Load checkpoint
checkpoint = checkpoint_manager.load_checkpoint(
    filename=f"model_epoch{epoch}.pt",
    experiment_name="my_experiment"
)

# Restore states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
checkpoint_manager.restore_rng_states(checkpoint)
```

## Troubleshooting

### Issue: "Checkpoint corrupted or incomplete"

**Cause:** Checkpoint save was interrupted (disk full, OOM, process killed)

**Solution:**
```python
# CheckpointManager automatically creates backups
# Check for .bak files in checkpoint directory
ls artifacts/checkpoints/*.pt.bak
```

### Issue: "Resume produces different results"

**Possible causes:**
1. Mid-epoch resume (see Known Limitation above)
2. Non-deterministic operations (e.g., atomic operations on GPU)
3. CUDA version mismatch between save and resume

**Solution:**
```bash
# Use deterministic mode
python run_all_kaggle.py --deterministic --resume

# Verify PyTorch version matches
python -c "import torch; print(torch.__version__)"
```

### Issue: "Checkpoint file too large"

**Cause:** Large models (ResNet-18 CIFAR-10 checkpoint ~45MB)

**Solution:**
```python
# Reduce max_checkpoints to save disk space
checkpoint_manager = CheckpointManager(
    checkpoint_dir="artifacts/checkpoints",
    max_checkpoints=2  # Keep only 2 most recent
)

# Or use compression (not implemented yet, planned for Phase 2)
```

## Future Improvements (Phase 2)

Planned enhancements for checkpoint/resume:

1. **DataLoader Worker RNG State Capture**
   - Save worker seeds and iteration positions
   - Enable true mid-epoch resume
   - Estimated effort: 200 lines, moderate complexity

2. **Checkpoint Compression**
   - Reduce checkpoint size by ~60% with torch.save(..., _use_new_zipfile_serialization=True)
   - Enable fast checkpoint uploads to cloud storage

3. **Async Checkpoint Writing**
   - Save checkpoints in background thread
   - Reduce training interruption from ~500ms to ~50ms

4. **Checkpoint Pruning Strategies**
   - Keep best N by validation accuracy
   - Keep exponentially spaced checkpoints (epoch 1, 2, 4, 8, ...)

## Related Documentation

- [README.md](../README.md): Quick start
- [docs/guides/EXPERIMENT_EXECUTION_GUIDE.md](guides/EXPERIMENT_EXECUTION_GUIDE.md): Running experiments
- [docs/OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md): Optional features

---

**Last Updated:** February 2, 2026  
**Maintainer:** GDSearch Team
