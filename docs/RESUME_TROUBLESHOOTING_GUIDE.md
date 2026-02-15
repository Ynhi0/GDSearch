# Resume Troubleshooting Guide

## 🎯 Quick Summary

Your experiments were starting from scratch because:
1. **Missing `--resume` flag** in command line
2. **No tuning cache** - hyperparameter tuning results weren't saved

## ✅ Fixes Implemented

### 1. Tuning Cache System (NEW)
- **File**: `src/core/tuning_cache.py` (created)
- **Purpose**: Save/load hyperparameter tuning results to avoid redundant Optuna studies
- **Integration**: Automatically used in `run_all_kaggle.py` MNIST experiments

### 2. Updated run_all_kaggle.py
- Added cache checking before tuning
- Saves tuning results after each optimizer
- Passes cache instance to all tuning calls

---

## 🚀 How to Use Resume Correctly

### Basic Resume (Skip Completed Experiments)
```bash
python run_all_kaggle.py \
  --experiments all \
  --seeds 42,123,456 \
  --results-dir /kaggle/working/results \
  --no-mlflow \
  --resume  # ← ADD THIS FLAG
```

### Resume with Tuning Cache (Recommended)
```bash
# First run: Tune + Train
python run_all_kaggle.py \
  --experiments mnist \
  --seeds 42 \
  --results-dir results \
  --no-mlflow

# Second run: Use cached tuning, skip completed
python run_all_kaggle.py \
  --experiments mnist \
  --seeds 123,456 \  # Different seeds
  --results-dir results \
  --no-mlflow \
  --resume
```

**What happens:**
- ✅ Tuning results loaded from cache (`results/tuning_cache/`)
- ✅ Completed seeds skipped (CSV exists)
- ✅ Only new seeds run with cached hyperparameters

### Skip Tuning Entirely
```bash
python run_all_kaggle.py \
  --experiments mnist \
  --seeds 42,123,456 \
  --results-dir results \
  --no-mlflow \
  --skip-tuning  # Uses default hyperparameters
```

---

## 📁 File Locations

### Training Results
```
results/
├── experiments/
│   └── mnist/
│       ├── MNIST_SimpleMLP_SGD_seed42.csv          # ← Resume checks this
│       ├── MNIST_SimpleMLP_SGD_seed42.metadata.json
│       └── ...
└── tuning_cache/                                    # ← NEW!
    ├── MNIST_SimpleMLP_SGD_tuned.json              # ← Cached tuning
    ├── MNIST_SimpleMLP_Adam_tuned.json
    └── ...
```

### Checkpoints
```
artifacts/
└── checkpoints/
    ├── MNIST_SGD_seed42.pt  # ← Resume from mid-training
    └── ...
```

---

## 🔍 How Resume Logic Works

### Level 1: Quick CSV Check
```python
if resume and is_experiment_completed(...):
    skip_experiment()
```
**Checks:** CSV file exists with ≥1 row

### Level 2: Checkpoint Resume
```python
action = decide_resume_action(checkpoint, ...)
if action == 'skip':
    continue
```
**Checks:** Checkpoint exists and has `training_complete=True`

### Level 3: Tuning Cache (NEW)
```python
cached_params = tuning_cache.load_tuned_params(...)
if cached_params:
    return cached_params
```
**Checks:** Tuning cache file exists for optimizer

---

## 🐛 Common Issues

### Issue: "Resume: Disabled" in logs
**Cause:** No `--resume` flag passed  
**Fix:** Add `--resume` to command

### Issue: Tuning runs every time
**Cause:** Tuning cache not implemented (FIXED)  
**Fix:** Now automatic with new cache system

### Issue: Experiments don't skip despite CSV existing
**Possible causes:**
1. CSV file path mismatch
2. CSV is empty (0 bytes)
3. `results_dir` structure different than expected

**Debug:**
```bash
# Check what files exist
find results/ -name "MNIST_*.csv"

# Check CSV content
head -n 5 results/experiments/mnist/MNIST_SimpleMLP_SGD_seed42.csv
```

### Issue: Cache not working
**Check cache directory:**
```bash
ls results/tuning_cache/
# Should see: MNIST_SimpleMLP_SGD_tuned.json, etc.
```

**Inspect cache file:**
```bash
cat results/tuning_cache/MNIST_SimpleMLP_SGD_tuned.json
```

Expected format:
```json
{
  "dataset": "MNIST",
  "model": "SimpleMLP",
  "optimizer": "SGD",
  "best_params": {"lr": 0.0868},
  "timestamp": "2026-02-03T14:52:32",
  "metadata": {
    "best_val_acc": 88.78,
    "n_trials": 15,
    "epochs": 3,
    "seed": 42
  }
}
```

---

## 🔧 Advanced Options

### Resume Behaviors
```bash
# Default: Skip if CSV exists
--resume --resume-behavior skip_if_results_exist

# Fail if checkpoint missing
--resume --resume-behavior error_if_no_checkpoint

# Restart from scratch if no checkpoint
--resume --resume-behavior restart_if_no_checkpoint
```

### Clear Tuning Cache
```python
from src.core.tuning_cache import create_tuning_cache

cache = create_tuning_cache("results")
cache.clear_cache(dataset="MNIST")  # Clear MNIST only
cache.clear_cache()  # Clear all
```

### Verify Resume Logic
```bash
python run_all_kaggle.py --verify-resume  # Runs golden test
```

---

## 📊 Expected Behavior

### First Run (No Cache)
```
INFO: Tuning SGD...
INFO:   Tuning SGD (15 trials, 3 epochs each)
[I] Trial 0 finished...
[I] Trial 1 finished...
...
INFO:     Best params: {'lr': 0.0868}
INFO: ✅ Saved tuning results to cache: MNIST_SimpleMLP_SGD_tuned.json
```

### Second Run (With Cache)
```
INFO: ✅ Using cached tuning results for SGD
INFO: Skipping SGD seed 42 (already completed)
```

---

## 🎓 Why This Matters

### Without Resume
- **Time wasted:** Re-running completed experiments
- **Compute cost:** Redundant GPU hours
- **Risk:** Accidentally overwriting good results

### Without Tuning Cache
- **15 trials × 3 epochs = 45 training epochs** per optimizer
- **12 optimizers × 45 = 540 epochs** wasted per re-run
- **10 seeds → 5,400 wasted epochs** total

### With Fixes
- ✅ Skip completed: Save hours of compute
- ✅ Cache tuning: Save thousands of epochs
- ✅ Safe restart: Never lose progress

---

## 📞 Still Having Issues?

1. **Enable debug logging:**
   ```bash
   python run_all_kaggle.py --resume --verbose 2>&1 | tee resume_debug.log
   ```

2. **Check file permissions:**
   ```bash
   ls -l results/tuning_cache/
   ```

3. **Verify imports:**
   ```bash
   python -c "from src.core.tuning_cache import TuningCache; print('OK')"
   ```

4. **Run minimal test:**
   ```bash
   python run_all_kaggle.py --experiments mnist --seeds 42 --ultra-quick --resume
   ```
