# Experiment Execution Guide for GDSearch

**Quick Reference for Running Experiments Independently**

---

## TL;DR - Quick Start

**Run single experiment:**
```bash
python run_all_kaggle.py --experiments mnist --quick --seeds 42,123,456
```

**Run multiple experiments:**
```bash
python run_all_kaggle.py --experiments mnist,cifar10 --quick --seeds 42,123
```

**Resume interrupted run:**
```bash
python run_all_kaggle.py --experiments mnist --resume --quick
```

**✅ All experiments are fully independent - run them in any order, any combination.**

---

## Table of Contents

1. [Independent Execution Patterns](#independent-execution-patterns)
2. [Running Single Experiments](#running-single-experiments)
3. [Running Multiple Experiments](#running-multiple-experiments)
4. [Parallel Execution](#parallel-execution)
5. [Resume and Checkpointing](#resume-and-checkpointing)
6. [Per-Optimizer Execution](#per-optimizer-execution)
7. [Seed Management](#seed-management)
8. [Common Workflows](#common-workflows)
9. [Troubleshooting](#troubleshooting)

---

## Independent Execution Patterns

### Core Principle

**Each experiment is fully independent:**
- ✅ Runs on its own dataset
- ✅ Saves to unique directory
- ✅ Uses independent checkpoints
- ✅ No cross-experiment dependencies

### Available Experiments

```bash
# Core experiments
mnist              # MNIST with SimpleMLP
cifar10            # CIFAR10 with ResNet-18
nlp                # IMDB sentiment with Transformers
medical            # Medical image segmentation with U-Net

# Optimization experiments
2d                 # 2D optimization landscapes
robustness         # Initial condition robustness
highdim            # High-dimensional optimization

# Ablation studies
ablation           # Component ablations
advanced_ablation  # Advanced training features
init_ablation      # Initialization methods
batch_ablation     # Batch size effects
lr_ablation        # Learning rate sensitivity
wd_ablation        # Weight decay impact
scheduler_ablation # Scheduler comparisons

# Analysis experiments
sam                # SAM sensitivity analysis
label_noise        # Label noise robustness
convergence_validation  # Convergence rate validation
theory_practice    # Theory-practice validation
```

---

## Running Single Experiments

### MNIST Experiment

**Quick test (2 epochs, 3 seeds):**
```bash
python run_all_kaggle.py --experiments mnist --quick --seeds 42,123,456
```

**Full experiment (default 50 epochs, 10 seeds):**
```bash
python run_all_kaggle.py --experiments mnist --seeds 42,123,456,789,1011,1213,1415,1617,1819,2021
```

**Ultra-quick test (2 epochs, 3 seeds):**
```bash
python run_all_kaggle.py --experiments mnist --ultra-quick --seeds 42,123,456
```

**Output:**
- Location: `results/experiments/mnist/`
- Files: `MNIST_SimpleMLP_{optimizer}_seed{N}.csv`
- Metadata: `MNIST_SimpleMLP_{optimizer}_seed{N}.metadata.json`

**Optimizers tested:**
- SGD, SGD_Momentum, Adam, AdamW, AMSGrad
- SAM_SGD, SAM_Adam, Lookahead_SGD, Lookahead_Adam
- AdaBound, RAdam, LAMB

### CIFAR-10 Experiment

**Quick test:**
```bash
python run_all_kaggle.py --experiments cifar10 --quick --seeds 42,123
```

**Full experiment:**
```bash
python run_all_kaggle.py --experiments cifar10 --seeds 42,123,456
```

**Output:**
- Location: `results/experiments/cifar10/`
- Files: `CIFAR10_ResNet18_{optimizer}_seed{N}.csv`

**Model:** ResNet-18 (11M parameters)  
**Data Augmentation:** Random crop, horizontal flip, normalization

### NLP Experiment

**Requirements:**
```bash
pip install transformers datasets torch
```

**Quick test:**
```bash
python run_all_kaggle.py --experiments nlp --quick --seeds 42
```

**Output:**
- Location: `results/experiments/nlp/`
- Files: `NLP_{model}_{optimizer}_seed{N}.csv`

**Models:** DistilBERT, BERT-base (if specified)

### 2D Optimization

**Quick test (fast convergence visualization):**
```bash
python run_all_kaggle.py --experiments 2d --quick --seeds 42
```

**Full experiment:**
```bash
python run_all_kaggle.py --experiments 2d --seeds 42,123,456
```

**Output:**
- Location: `results/experiments/2d_optimization/`
- Functions: Rosenbrock, Rastrigin, Ackley, SaddlePoint

---

## Running Multiple Experiments

### Sequential Execution (Default)

**Run MNIST and CIFAR10:**
```bash
python run_all_kaggle.py --experiments mnist,cifar10 --quick --seeds 42,123
```

**Execution order:** MNIST → CIFAR10 (sequential)

**Run all core experiments:**
```bash
python run_all_kaggle.py --experiments mnist,cifar10,nlp,2d --quick
```

### Order Independence

**Experiments can run in ANY order:**

```bash
# Order 1: MNIST → CIFAR10
python run_all_kaggle.py --experiments mnist,cifar10

# Order 2: CIFAR10 → MNIST (reversed, equally valid)
python run_all_kaggle.py --experiments cifar10,mnist

# Order 3: Random selection
python run_all_kaggle.py --experiments 2d,nlp,mnist
```

**All produce identical results** (given same seeds).

---

## Parallel Execution

### ⚠️ Current Limitation

**Default implementation is sequential.** Experiments run one after another.

### Safe Parallel Execution

**Method 1: Multiple Terminal Windows**

```bash
# Terminal 1: MNIST
python run_all_kaggle.py --experiments mnist --quick --seeds 42,123 &

# Terminal 2: CIFAR10
python run_all_kaggle.py --experiments cifar10 --quick --seeds 42,123 &

# Terminal 3: 2D Optimization
python run_all_kaggle.py --experiments 2d --quick --seeds 42,123 &
```

**Safety:**
- ✅ Unique output directories (no file conflicts)
- ✅ Independent datasets
- ⚠️ **GPU Memory:** May cause OOM if all on same GPU

### Method 2: Assign Different GPUs

**If you have multiple GPUs:**

```bash
# Terminal 1: MNIST on GPU 0
CUDA_VISIBLE_DEVICES=0 python run_all_kaggle.py --experiments mnist --quick &

# Terminal 2: CIFAR10 on GPU 1
CUDA_VISIBLE_DEVICES=1 python run_all_kaggle.py --experiments cifar10 --quick &

# Terminal 3: NLP on CPU
CUDA_VISIBLE_DEVICES="" python run_all_kaggle.py --experiments nlp --quick &
```

**Recommended for:** Large-scale experiments with multiple GPUs

### Method 3: Time-Shifted Execution

**Start experiments at different times:**

```bash
# Start MNIST now
python run_all_kaggle.py --experiments mnist --time-budget 2.0 &

# Start CIFAR10 in 2 hours (after MNIST completes)
sleep 7200 && python run_all_kaggle.py --experiments cifar10 &
```

**Use case:** Single GPU, sequential execution with automatic queuing

---

## Resume and Checkpointing

### Basic Resume

**Command:**
```bash
python run_all_kaggle.py --experiments mnist --resume --quick
```

**Behavior:**
- Skips experiments that already have result CSV files
- Checks: `results/experiments/mnist/MNIST_SimpleMLP_{opt}_seed{N}.csv`
- If exists and has data → skip
- If missing or empty → run

### Resume Modes

**Skip if results exist (default with --resume):**
```bash
python run_all_kaggle.py --experiments mnist --resume --resume-behavior skip_if_results_exist
```

**Restart from checkpoint:**
```bash
python run_all_kaggle.py --experiments mnist --resume --resume-behavior restart_if_no_checkpoint
```

**Error if no checkpoint:**
```bash
python run_all_kaggle.py --experiments mnist --resume --resume-behavior error_if_no_checkpoint
```

### Checkpoint Files

**Location:** `artifacts/checkpoints/`

**Format:**
```
MNIST_SGD_seed42.pt
MNIST_Adam_seed123.pt
CIFAR10_ResNet18_SGD_seed42.pt
```

**Contents:**
- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Training history
- RNG states (for reproducibility)
- Metadata (epoch, completed flag)

### Resume Workflow

**Scenario:** Experiment crashed after 30 minutes

```bash
# 1. Check what completed
ls results/experiments/mnist/*.csv

# 2. Resume with --resume flag
python run_all_kaggle.py --experiments mnist --resume --seeds 42,123,456

# Output:
# "Skipping SGD seed 42 (already completed)"
# "Skipping Adam seed 42 (already completed)"
# "Running AdamW seed 42..."  # Continues from here
```

### Partial Completion

**Resume respects per-(optimizer, seed) completion:**

```bash
# Initially ran:
python run_all_kaggle.py --experiments mnist --seeds 42,123 --quick
# (Completed: SGD seed 42, SGD seed 123, Adam seed 42)
# (Crashed before: Adam seed 123, AdamW seed 42, ...)

# Resume:
python run_all_kaggle.py --experiments mnist --resume --seeds 42,123 --quick
# Skips: SGD seed 42, SGD seed 123, Adam seed 42
# Runs: Adam seed 123, AdamW seed 42, AdamW seed 123, ...
```

---

## Per-Optimizer Execution

### Running Specific Optimizers

**Currently:** All optimizers run by default

**To filter optimizers, edit code:**

**File:** [run_all_kaggle.py](run_all_kaggle.py) line 2853 (MNIST) or 3631 (CIFAR10)

**Before:**
```python
optimizers_config = [
    ('Adam', 0.001),
    ('AdamW', 0.001),
    ('SGD_Momentum', 0.01),
    # ... all optimizers
]
```

**After (filter to specific optimizers):**
```python
optimizers_config = [
    ('Adam', 0.001),
    ('AdamW', 0.001),
    # Only Adam and AdamW
]
```

**Run:**
```bash
python run_all_kaggle.py --experiments mnist --quick
```

### Per-Optimizer Resume

**Scenario:** Want to rerun only SGD

```bash
# 1. Delete SGD results
rm results/experiments/mnist/MNIST_SimpleMLP_SGD_seed*.csv

# 2. Resume (will skip others, rerun SGD)
python run_all_kaggle.py --experiments mnist --resume --quick
```

---

## Seed Management

### Default Seeds

**Default:** 10 seeds for statistical validity
```python
--seeds 42,123,456,789,1011,1213,1415,1617,1819,2021
```

### Custom Seeds

**Quick test (3 seeds):**
```bash
python run_all_kaggle.py --experiments mnist --quick --seeds 42,123,456
```

**Minimal test (1 seed):**
```bash
python run_all_kaggle.py --experiments mnist --ultra-quick --seeds 42
```

**High-confidence experiment (20 seeds):**
```bash
python run_all_kaggle.py --experiments mnist \
  --seeds 42,123,456,789,1011,1213,1415,1617,1819,2021,2223,2425,2627,2829,3031,3233,3435,3637,3839,4041
```

### Seed Independence

**Each seed runs independently:**

```bash
# Run seed 42 only
python run_all_kaggle.py --experiments mnist --seeds 42

# Later, add more seeds
python run_all_kaggle.py --experiments mnist --seeds 123,456 --resume
# (Skips seed 42, runs 123 and 456)
```

### Reproducibility

**Same seed → same results:**

```bash
# Run 1
python run_all_kaggle.py --experiments mnist --seeds 42 --quick
# Result: test_acc = 97.34%

# Run 2 (identical command)
python run_all_kaggle.py --experiments mnist --seeds 42 --quick
# Result: test_acc = 97.34% (identical)
```

**Guaranteed by:**
- Deterministic random seed initialization
- Checkpoint includes RNG states
- No random file ordering

---

## Common Workflows

### Workflow 1: Quick Validation

**Goal:** Test experiment pipeline quickly

```bash
# Ultra-quick mode: 2 epochs, 1 seed
python run_all_kaggle.py --experiments mnist --ultra-quick --seeds 42
```

**Expected time:** ~2 minutes  
**Use case:** CI/CD, quick validation

### Workflow 2: Local Development

**Goal:** Test changes before full run

```bash
# Quick mode: 5-20 epochs, 3 seeds
python run_all_kaggle.py --experiments mnist,cifar10 --quick --seeds 42,123,456
```

**Expected time:** ~30 minutes  
**Use case:** Development, debugging

### Workflow 3: Full Reproducible Experiment

**Goal:** Publication-quality results

```bash
# Full mode: 50 epochs, 10 seeds, deterministic
python run_all_kaggle.py \
  --experiments mnist,cifar10 \
  --seeds 42,123,456,789,1011,1213,1415,1617,1819,2021 \
  --deterministic
```

**Expected time:** ~8 hours  
**Use case:** Research papers, benchmarks

### Workflow 4: Kaggle Competition

**Goal:** Optimize for Kaggle T4 GPU (11-hour limit)

```bash
# Kaggle-optimized: larger batches, mixed precision
python run_all_kaggle.py \
  --experiments mnist,cifar10,2d \
  --kaggle-t4 \
  --quick \
  --time-budget 11.0 \
  --results-dir /kaggle/working/results
```

**Features:**
- `--kaggle-t4`: Optimized batch sizes (256-512)
- `--time-budget 11.0`: Graceful exit before timeout
- Mixed precision (AMP) enabled automatically

### Workflow 5: Incremental Experiment Addition

**Goal:** Add new experiments to existing results

```bash
# Day 1: Run MNIST
python run_all_kaggle.py --experiments mnist --quick

# Day 2: Add CIFAR10 (without rerunning MNIST)
python run_all_kaggle.py --experiments mnist,cifar10 --resume --quick
# (Skips MNIST, runs only CIFAR10)

# Day 3: Add NLP
python run_all_kaggle.py --experiments mnist,cifar10,nlp --resume --quick
# (Skips MNIST and CIFAR10, runs only NLP)
```

**Use case:** Gradual experiment accumulation

### Workflow 6: Selective Rerun

**Goal:** Rerun specific experiments after bug fix

```bash
# 1. Delete affected results
rm -rf results/experiments/mnist/

# 2. Rerun only MNIST (keep CIFAR10 and NLP)
python run_all_kaggle.py --experiments mnist,cifar10,nlp --resume
# (Runs MNIST, skips CIFAR10 and NLP)
```

---

## Troubleshooting

### Problem 1: "Already completed" message but no results

**Symptom:**
```
Skipping MNIST SGD seed 42 (already completed)
```
But CSV file is empty or corrupted.

**Solution:**
```bash
# Delete corrupted file
rm results/experiments/mnist/MNIST_SimpleMLP_SGD_seed42.csv

# Rerun with --resume
python run_all_kaggle.py --experiments mnist --resume --quick
```

### Problem 2: Out of GPU memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution 1: Reduce batch size**
```bash
python run_all_kaggle.py --experiments mnist --quick
# (Automatically reduces batch size on OOM)
```

**Solution 2: Use Kaggle T4 optimizations**
```bash
python run_all_kaggle.py --experiments mnist --kaggle-t4 --quick
```

**Solution 3: Run on CPU**
```bash
CUDA_VISIBLE_DEVICES="" python run_all_kaggle.py --experiments mnist --quick
```

### Problem 3: Experiment takes too long

**Symptom:** Experiment running longer than expected

**Solution 1: Use time budget**
```bash
python run_all_kaggle.py --experiments mnist --time-budget 2.0 --quick
# (Exits gracefully after 2 hours)
```

**Solution 2: Reduce seeds**
```bash
python run_all_kaggle.py --experiments mnist --seeds 42,123 --quick
# (Fewer seeds = faster)
```

**Solution 3: Ultra-quick mode**
```bash
python run_all_kaggle.py --experiments mnist --ultra-quick --seeds 42
# (2 epochs only)
```

### Problem 4: Cannot find results

**Symptom:** Results not in expected location

**Check:**
```bash
# Default location
ls results/experiments/mnist/

# Custom location (if --results-dir used)
ls /kaggle/working/results/experiments/mnist/
```

**Find all results:**
```bash
find results/ -name "*.csv" -type f
```

### Problem 5: Checkpoint conflicts

**Symptom:** Resume doesn't work correctly

**Solution 1: Clear checkpoints**
```bash
rm -rf artifacts/checkpoints/MNIST_*.pt
python run_all_kaggle.py --experiments mnist --quick
```

**Solution 2: Use resume behavior flag**
```bash
python run_all_kaggle.py --experiments mnist --resume --resume-behavior restart_if_no_checkpoint
```

### Problem 6: MLflow tracking errors

**Symptom:**
```
MLflow tracking failed: ...
```

**Solution: Disable MLflow**
```bash
python run_all_kaggle.py --experiments mnist --no-mlflow --quick
```

---

## Advanced Usage

### Hyperparameter Tuning

**Skip tuning (use defaults):**
```bash
python run_all_kaggle.py --experiments mnist --skip-tuning --quick
```

**Enable tuning (default):**
```bash
python run_all_kaggle.py --experiments mnist --quick
# (Runs Optuna tuning: 15 trials, 3 epochs per trial)
```

**Note:** Tuning is per-experiment and doesn't affect other experiments.

### Mixed Precision Training

**Enable AMP (automatic mixed precision):**
```bash
python run_all_kaggle.py --experiments mnist --use-amp --quick
```

**Benefits:**
- 2× faster training
- 50% less GPU memory
- Minimal accuracy loss

### Exponential Moving Average (EMA)

**Enable EMA:**
```bash
python run_all_kaggle.py --experiments mnist --use-ema --quick
```

**Benefits:**
- Better generalization
- Smoother weight updates
- +0.5% accuracy typical

### Label Smoothing

**Enable label smoothing:**
```bash
python run_all_kaggle.py --experiments mnist --label-smoothing 0.1 --quick
```

**Typical value:** 0.1 (10% smoothing)

### Profiling

**Enable performance profiling:**
```bash
python run_all_kaggle.py --experiments mnist --profile --quick
```

**Output:**
- GPU utilization
- Memory usage
- Training speed (samples/sec)

### Generate Final Reports

**After experiments complete:**
```bash
python run_all_kaggle.py --experiments mnist,cifar10 --quick --generate-deliverables
```

**Generates:**
- High-quality plots
- Statistical reports
- Summary tables
- Cross-experiment analysis

---

## Quick Reference Table

| Goal | Command |
|------|---------|
| **Quick test (single exp)** | `python run_all_kaggle.py --experiments mnist --quick --seeds 42` |
| **Multiple experiments** | `python run_all_kaggle.py --experiments mnist,cifar10 --quick` |
| **Resume interrupted** | `python run_all_kaggle.py --experiments mnist --resume --quick` |
| **Ultra-fast test** | `python run_all_kaggle.py --experiments mnist --ultra-quick --seeds 42` |
| **Full experiment** | `python run_all_kaggle.py --experiments mnist` |
| **Kaggle-optimized** | `python run_all_kaggle.py --experiments mnist --kaggle-t4 --time-budget 11` |
| **Skip tuning** | `python run_all_kaggle.py --experiments mnist --skip-tuning --quick` |
| **CPU only** | `CUDA_VISIBLE_DEVICES="" python run_all_kaggle.py --experiments mnist --quick` |
| **Single GPU** | `CUDA_VISIBLE_DEVICES=0 python run_all_kaggle.py --experiments mnist --quick` |
| **Disable MLflow** | `python run_all_kaggle.py --experiments mnist --no-mlflow --quick` |

---

## Summary

✅ **Experiments are fully independent**  
✅ **Run in any order**  
✅ **Resume safely**  
✅ **No cross-dependencies**  
✅ **Parallel execution supported (with GPU management)**

**For more details, see:** [EXPERIMENT_INDEPENDENCE_ANALYSIS.md](EXPERIMENT_INDEPENDENCE_ANALYSIS.md)

---

**Last Updated:** February 2, 2026  
**Version:** 1.0
