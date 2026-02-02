# Kaggle T4x2 Integration Guide

## Quick Start: Enable Parallel Experiments on Kaggle T4x2

### Step 1: Add GPU Detection Cell (After Setup)

```python
# Cell: GPU Configuration Detection
import torch
from src.utils.parallel_experiment_runner import detect_gpu_configuration

print("="*80)
print("GPU CONFIGURATION DETECTION")
print("="*80)

gpu_config = detect_gpu_configuration()

print(f"\n✓ GPU Count: {gpu_config['gpu_count']}")
if gpu_config['gpu_count'] > 0:
    print(f"\nGPU Details:")
    for i, (name, mem_gb) in enumerate(zip(gpu_config['gpu_names'], gpu_config['gpu_memory'])):
        print(f"  GPU {i}: {name}")
        print(f"         Memory: {mem_gb:.2f} GB")

print(f"\n✓ Parallel Execution Capable: {gpu_config['parallel_capable']}")
print(f"✓ Parallel Execution Recommended: {gpu_config['recommended_parallel']}")

if gpu_config['recommended_parallel']:
    print(f"\n🚀 MULTI-GPU DETECTED!")
    print(f"   Expected speedup: ~{gpu_config['gpu_count']}x for independent experiments")
    print(f"   Parallel mode will be ENABLED")
    PARALLEL_MODE = True
    NUM_GPUS = gpu_config['gpu_count']
else:
    print(f"\nℹ️  Single GPU or CPU mode")
    print(f"   Parallel mode will be DISABLED")
    PARALLEL_MODE = False
    NUM_GPUS = 1

print("="*80)
```

### Step 2: Update Experiment Execution Cell

```python
# Cell: Run Experiments (with Parallel Support)
import sys
import subprocess
import logging

# Build base command
cmd = [
    sys.executable,
    'run_all_kaggle.py',
    '--experiments', 'mnist_sgd,mnist_adam,cifar10_sgd,cifar10_adam',
    '--seeds', '42,123,456',
    '--epochs', '50',
    '--results-dir', 'results',
    '--no-mlflow'  # Disable MLflow for faster runs
]

# Add parallel execution if available
if PARALLEL_MODE:
    cmd.extend(['--parallel', '--num-gpus', str(NUM_GPUS)])
    print(f"🚀 PARALLEL MODE: Running experiments on {NUM_GPUS} GPUs simultaneously")
    print(f"   Expected speedup: ~{NUM_GPUS}x")
else:
    print("Sequential mode: Single GPU/CPU")

# Add resume support (skip completed experiments)
cmd.append('--resume')
print(f"✓ Resume mode enabled: Will skip completed experiments")

print(f"\nExecuting command:")
print(' '.join(cmd))
print("="*80)

# Run experiments
result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode == 0:
    print("\n✅ Experiments completed successfully!")
else:
    print(f"\n❌ Experiments failed with exit code {result.returncode}")
```

### Step 3: Add Progress Monitoring Cell (Optional)

```python
# Cell: Check Experiment Progress
from pathlib import Path
from src.utils.resume_utils import count_completed_experiments

# Define experiments to check
all_experiments = [
    {'name': f'mnist_sgd_seed{seed}', 'model': 'SimpleMLP', 'dataset': 'MNIST', 
     'optimizer': 'SGD', 'lr': 0.01, 'seed': seed, 'epochs': 50}
    for seed in [42, 123, 456]
] + [
    {'name': f'mnist_adam_seed{seed}', 'model': 'SimpleMLP', 'dataset': 'MNIST',
     'optimizer': 'Adam', 'lr': 0.001, 'seed': seed, 'epochs': 50}
    for seed in [42, 123, 456]
]

results_dir = Path('results')
stats = count_completed_experiments(all_experiments, results_dir, expected_epochs=50)

print("="*80)
print("EXPERIMENT PROGRESS")
print("="*80)
print(f"✓ Completed: {stats['completed']}")
print(f"⏳ Incomplete: {stats['incomplete']}")
print(f"📊 Total: {stats['total']}")
print(f"Progress: {stats['completed']/stats['total']*100:.1f}%")
print("="*80)
```

---

## CLI Arguments Reference

### Parallel Execution
```bash
python run_all_kaggle.py --parallel --num-gpus 2
```
- `--parallel`: Enable parallel execution across multiple GPUs
- `--num-gpus N`: Use N GPUs (default: auto-detect all available)

### Resume Support
```bash
python run_all_kaggle.py --resume
```
- `--resume`: Skip experiments that have already completed successfully
- Validates result files (checks epochs, columns, no corruption)
- Safe: Re-runs incomplete or corrupted experiments

### Combined Usage
```bash
python run_all_kaggle.py \
    --parallel \
    --num-gpus 2 \
    --resume \
    --experiments mnist_sgd,mnist_adam \
    --seeds 42,123,456 \
    --epochs 50
```

---

## Expected Performance on Kaggle T4x2

### Without Parallel Mode (Sequential)
- **1 experiment**: ~5 minutes (50 epochs)
- **10 experiments**: ~50 minutes
- **100 experiments**: ~8.3 hours

### With Parallel Mode (2 GPUs)
- **1 experiment**: ~5 minutes (same, single experiment)
- **10 experiments**: ~25 minutes (2x speedup)
- **100 experiments**: ~4.2 hours (2x speedup)

**Speedup Formula**: `speedup ≈ min(num_experiments / num_gpus, num_gpus)`

---

## Troubleshooting

### Issue: "Parallel mode not available" on T4x2
**Cause**: Kaggle might not expose both GPUs to Python

**Solution**:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Explicitly enable both GPUs
```

### Issue: CUDA out of memory with parallel mode
**Cause**: Both GPUs trying to use full memory

**Solution**:
```python
# Reduce batch size in config
config['batch_size'] = 64  # Instead of 128
```

Or use `--max-memory` flag:
```bash
python run_all_kaggle.py --parallel --max-memory 3.5  # GB per GPU
```

### Issue: One experiment fails, stops all others
**Cause**: Error handling not catching exception

**Solution**: Parallel runner already handles this. Check logs:
```python
# Results include both success and error status
for result in results:
    if result['status'] == 'error':
        print(f"Failed: {result['experiment']}: {result['error']}")
```

---

## Performance Tips

### 1. Use Resume Mode for Long Runs
```bash
# First run (interrupted)
python run_all_kaggle.py --experiments all --parallel

# Resume (skips completed)
python run_all_kaggle.py --experiments all --parallel --resume
```

### 2. Disable MLflow for Speed
```bash
python run_all_kaggle.py --parallel --no-mlflow
```
MLflow tracking adds ~10% overhead. Disable for faster iteration.

### 3. Use Quick Mode for Testing
```bash
python run_all_kaggle.py --parallel --quick --seeds 42
```
Runs with reduced epochs/batch size to verify setup.

### 4. Optimize Experiment Order
Place fast experiments first to verify pipeline:
```bash
python run_all_kaggle.py \
    --experiments mnist_sgd,mnist_adam \  # Fast (SimpleMLP)
    --parallel
    
# Then run slower experiments
python run_all_kaggle.py \
    --experiments cifar10_resnet,imagenet_vit \  # Slow
    --parallel --resume
```

---

## Checkpoint Management

### Enable Checkpoint Saving
```python
# In your experiment config
config['save_checkpoints'] = True
config['checkpoint_dir'] = 'checkpoints'
config['keep_last'] = 3
config['keep_best'] = 3
```

### Resume from Checkpoint
```python
from src.utils.checkpoint_utils import load_checkpoint_safe

checkpoint_path = Path('checkpoints/best_checkpoint.pt')
if checkpoint_path.exists():
    metadata = load_checkpoint_safe(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device
    )
    start_epoch = metadata['epoch'] + 1
    print(f"Resumed from epoch {metadata['epoch']}")
```

---

## Validation

### Test Parallel Speedup
```python
import time

# Sequential
start = time.time()
run_experiments_sequential(experiments[:10])
seq_time = time.time() - start

# Parallel
start = time.time()
runner = ParallelExperimentRunner(num_gpus=2)
runner.run_experiments_parallel(experiments[:10])
par_time = time.time() - start

speedup = seq_time / par_time
print(f"Speedup: {speedup:.2f}x (expected: ~2x)")
```

### Verify GPU Utilization
```bash
# In separate terminal
watch -n 1 nvidia-smi
```
You should see both GPUs active (~80-100% utilization) during parallel execution.

---

## Example: Complete Kaggle Notebook Workflow

```python
# 1. Setup
!pip install -r requirements.txt

# 2. Detect GPUs
from src.utils.parallel_experiment_runner import detect_gpu_configuration
gpu_config = detect_gpu_configuration()
PARALLEL = gpu_config['recommended_parallel']

# 3. Define experiments
experiments = [
    'mnist_sgd', 'mnist_adam', 'mnist_rmsprop',
    'cifar10_sgd', 'cifar10_adam', 'cifar10_rmsprop'
]
seeds = [42, 123, 456, 789, 999]

# 4. Run with parallel + resume
cmd = [
    'python', 'run_all_kaggle.py',
    '--experiments', ','.join(experiments),
    '--seeds', ','.join(map(str, seeds)),
    '--epochs', '50',
    '--resume'
]

if PARALLEL:
    cmd.extend(['--parallel', '--num-gpus', str(gpu_config['gpu_count'])])

!{' '.join(cmd)}

# 5. Check progress
from src.utils.resume_utils import count_completed_experiments
stats = count_completed_experiments(all_experiments, Path('results'))
print(f"Progress: {stats['completed']}/{stats['total']}")

# 6. Collect results
import pandas as pd
results = []
for exp in experiments:
    for seed in seeds:
        result_file = f"results/{exp}_seed{seed}.csv"
        if Path(result_file).exists():
            df = pd.read_csv(result_file)
            results.append(df)

final_df = pd.concat(results, ignore_index=True)
print(final_df.groupby(['optimizer'])['test_acc'].mean())
```

---

## FAQ

**Q: Will parallel mode work on Kaggle P100 or T4x1?**
A: No, you need 2+ GPUs. The system automatically falls back to sequential mode.

**Q: Can I mix CPU and GPU experiments in parallel?**
A: Not currently. All experiments run on GPU if available, CPU if not.

**Q: Does resume mode check if hyperparameters changed?**
A: No, it only checks if result file exists and is complete. If you change hyperparameters, delete old results first.

**Q: Can I use parallel mode with Optuna hyperparameter tuning?**
A: Yes, but better to parallelize Optuna trials, not individual experiments.

---

**Integration Status**: Ready for deployment
**Testing Required**: Measure speedup on actual Kaggle T4x2
**Expected ROI**: 2x faster benchmarks, saving hours per run
