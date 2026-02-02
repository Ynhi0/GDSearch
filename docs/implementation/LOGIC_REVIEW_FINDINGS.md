# GDSearch Logic Review - Critical Findings

**Reviewed by:** Deep Logic Analysis - Research Analyst Mode  
**Date:** February 1, 2026  
**Scope:** Data pipeline, experiment orchestration, result processing

---

## CRITICAL ISSUE #1: Data Augmentation Leakage in Validation Splits

### **Location:** `src/runners/data_loading.py` lines 90-130

### **Problem:**
Validation and test sets are being created using `Subset()` on an **augmented training dataset**, causing data augmentation to leak into validation/test evaluation.

```python
# CIFAR10 example (MNIST has same issue):
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     # ← AUGMENTATION
    transforms.RandomHorizontalFlip(),        # ← AUGMENTATION
    transforms.ToTensor(),
    transforms.Normalize(...)
])

train_dataset = datasets.CIFAR10('./data', train=True, transform=transform_train)

# ❌ BUG: val_subset inherits transform_train (with augmentation!)
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)  # ← USES AUGMENTED TRANSFORMS!
```

### **Impact:**
- **Validation metrics are artificially inflated** because random crops/flips create easier samples
- **Hyperparameter tuning is biased** - choosing HPs that work well on augmented (easier) validation data
- **Test set contamination** if any code path creates test splits from training datasets
- **Irreproducible results** - validation accuracy varies across runs due to random augmentations

### **Root Cause:**
When you create a `Subset` of a `Dataset`, the subset **inherits the parent's transform**. So `val_subset` gets `RandomCrop` and `RandomHorizontalFlip` applied during evaluation, which is incorrect.

### **Fix Required:**
Create separate datasets with different transforms:

```python
# CORRECT approach:
train_transform = transforms.Compose([...with augmentation...])
eval_transform = transforms.Compose([...NO augmentation...])

# Load raw data
train_data_raw = datasets.CIFAR10('./data', train=True, download=True)
test_data = datasets.CIFAR10('./data', train=False, transform=eval_transform)

# Split indices
train_indices, val_indices = split_indices(len(train_data_raw), val_split, seed)

# Create subsets with CORRECT transforms
train_subset = TransformedSubset(train_data_raw, train_indices, train_transform)
val_subset = TransformedSubset(train_data_raw, val_indices, eval_transform)  # No augmentation!
```

---

## CRITICAL ISSUE #2: Resume Logic Path Confusion

### **Location:** `run_all_kaggle.py` line 1069 (`is_experiment_completed`) and line 1303 (`save_run_artifacts`)

### **Problem:**
The resume detection logic tries to handle multiple calling conventions but creates **path nesting bugs**:

```python
# Caller passes: "results/"
# Function creates: "results/experiments/mnist" ✓ CORRECT

# Caller passes: "results/experiments/mnist" (already nested)
# Function creates: "results/experiments/mnist/mnist" ✗ WRONG!
```

### **Impact:**
- Resume fails silently - thinks experiments aren't complete when they are
- Results scatter across multiple directories
- Metadata files become orphaned
- Summary CSV aggregation breaks

### **Root Cause:**
Defensive path handling creates double-nesting when callers are inconsistent:

```python
if results_dir.name.lower() == dataset.lower():
    results_base = results_dir  # Already at mnist/
elif "experiments" in [p.lower() for p in results_dir.parts]:
    results_base = results_dir / dataset.lower()  # Add mnist again!
else:
    results_base = results_dir / "experiments" / dataset.lower()
```

### **Fix Required:**
**Enforce canonical paths at entry point:**

```python
# In run_mnist_experiment, run_cifar10_experiment, etc:
def run_mnist_experiment(results_dir="results", ...):
    # Normalize at entry - always store relative to results/
    results_base = Path(results_dir) / "experiments" / "mnist"
    results_base.mkdir(parents=True, exist_ok=True)
    
    # Pass normalized path everywhere
    is_experiment_completed(results_base, 'MNIST', ...)
    save_run_artifacts(results_base, 'MNIST', ...)
```

---

## CRITICAL ISSUE #3: Seed Isolation Failure in Multi-Seed Runs

### **Location:** `run_all_kaggle.py` lines 2940-2950 (MNIST experiment loop)

### **Problem:**
Seeds are set correctly, but **model weights/optimizer state may leak across runs** if exceptions occur:

```python
for seed in seeds:
    set_seed(seed)
    model = SimpleMLP().to(device)  # ✓ New model
    optimizer = opt_func(model.parameters())  # ✓ New optimizer
    
    try:
        train_loop(...)
    except Exception:
        continue  # ❌ Model stays in GPU memory with seed N state
                  # Next iteration (seed N+1) might reuse contaminated state
```

### **Impact:**
- **Cross-seed contamination** if OOM or other exception occurs
- **Non-reproducible multi-seed runs**
- **Statistic aggregation errors** (means/stds computed on non-independent runs)

### **Fix Required:**
```python
for seed in seeds:
    try:
        # Clear all state at start of each seed
        clear_gpu_memory(force=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        set_seed(seed)
        model = SimpleMLP().to(device)
        optimizer = opt_func(model.parameters())
        
        train_loop(...)
    finally:
        # Guaranteed cleanup even on exception
        del model, optimizer
        clear_gpu_memory()
```

---

## CRITICAL ISSUE #4: CSV Read Safety vs Atomic Write Mismatch

### **Location:** `src/utils/csv_utils.py`

### **Problem:**
`safe_read_csv` is defensive about corruption, but writes aren't atomic:

```python
# Writing (no atomicity):
df.to_csv(path, index=False)  # ❌ Can be corrupted mid-write if crash/OOM

# Reading (overly defensive):
if size == 0: return None
if sample.columns is None: return None  # ← Treats partial writes as "empty"
```

### **Impact:**
- **Data loss on crashes** - partial CSVs treated as "corrupted" and moved to quarantine
- **Resume logic fails** - incomplete CSVs from interrupted runs look "empty"
- **Results accumulation breaks** - legitimate partial results discarded

### **Fix Required:**
```python
def safe_write_csv(df, path, **kwargs):
    """Atomic CSV write with temp file + rename."""
    path = Path(path)
    temp_path = path.with_suffix('.csv.tmp')
    
    try:
        df.to_csv(temp_path, index=False, **kwargs)
        temp_path.replace(path)  # Atomic on POSIX, near-atomic on Windows
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
```

---

## CRITICAL ISSUE #5: Metric Naming Inconsistency

### **Location:** Multiple files - `run_all_kaggle.py`, `src/analysis/*.py`, `src/visualization/*.py`

### **Problem:**
Results use inconsistent metric names across experiments:

```python
# MNIST might save:
{'final_test_acc': 95.2, 'test_accuracy': 94.8, ...}

# CIFAR10 might save:
{'test_acc': 92.1, 'final_accuracy': 91.5, ...}

# NLP might save:
{'accuracy': 89.3, 'eval_acc': 88.9, ...}
```

### **Impact:**
- **Plotting scripts break** when they expect 'test_acc' but find 'test_accuracy'
- **Statistical analysis fails** with KeyError
- **Result aggregation produces NaNs**

### **Current Mitigation:**
`normalize_metric_names()` exists (line 318) but **isn't called consistently**

### **Fix Required:**
```python
# Enforce at save time:
def save_run_artifacts(..., history, ...):
    # Normalize BEFORE saving
    normalized_history = [normalize_metric_names(row) for row in history]
    df = pd.DataFrame(normalized_history)
    safe_to_csv(df, csv_path)
```

---

## CRITICAL ISSUE #6: Gradient Accumulation Arithmetic Error

### **Location:** Training loops throughout codebase (e.g., `src/experiments/run_nn_experiment.py`)

### **Problem:**
When using gradient accumulation, the code doesn't scale the loss:

```python
loss = criterion(outputs, targets)
loss.backward()  # ❌ Gradients accumulate over batches

if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### **Impact:**
- **Effective learning rate is multiplied by accumulation_steps**
- **Non-reproducible results** when changing batch size
- **Invalidates all hyperparameter tuning** done without accumulation

### **Fix Required:**
```python
loss = criterion(outputs, targets)
loss = loss / accumulation_steps  # ← CRITICAL: Scale loss
loss.backward()

if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## ISSUE #7: DataLoader Worker Seed Determinism

### **Location:** `run_all_kaggle.py` line 1421 (`_worker_init`)

### **Problem:**
Worker init function is defined but **not all DataLoader calls use it**:

```python
# Some loaders use it:
make_dataloader(..., seed=42)  # ✓ Uses worker_init_fn

# Others don't:
DataLoader(dataset, batch_size=128, shuffle=True)  # ✗ Non-deterministic workers
```

### **Impact:**
- **Random data ordering** in workers breaks reproducibility
- **Validation metrics vary** across identical runs
- **Multi-GPU training non-deterministic**

### **Fix Required:**
Audit all `DataLoader(...)` calls and replace with `make_dataloader(..., seed=seed)`

---

## ISSUE #8: Resume Behavior Inconsistency

### **Location:** Multiple experiment runners

### **Problem:**
Some experiments check `if resume and is_experiment_completed(...)` but the behavior varies:

```python
# MNIST: Skip optimizer entirely
if resume and is_experiment_completed(...):
    continue

# CIFAR10: Skip seed within optimizer  
for seed in seeds:
    if resume and is_experiment_completed(...):
        continue

# NLP: No resume check at all!
```

### **Impact:**
- **Incomplete resumes** - might skip whole optimizers or just seeds
- **Wasted computation** - re-runs completed work
- **Inconsistent metadata** - can't tell what was resumed vs re-run

### **Fix Required:**
Standardize resume logic:

```python
def should_skip_experiment(results_dir, dataset, model, optimizer, seed, resume):
    """Centralized resume decision."""
    if not resume:
        return False
    return is_experiment_completed(results_dir, dataset, model, optimizer, seed)
```

---

## ISSUE #9: Test Set Usage in Hyperparameter Tuning

### **Location:** `run_all_kaggle.py` line 1896 (quick_tune_optimizer)

### **Problem:**
The docstring says to use validation data, but some callers **might pass test loaders**:

```python
def quick_tune_optimizer(..., val_loader, ...):
    """
    Args:
        val_loader: VALIDATION DataLoader (NOT test set!)
    """
    # ✓ Has defensive checks, but depends on loader having correct metadata
```

### **Impact:**
- **Test set leakage** if caller passes wrong loader
- **Overfitting to test distribution**
- **Invalid generalization claims**

### **Current Mitigation:**
Function checks `loader.name` and `_split_type` attributes, but **not all loaders have these**

### **Fix Required:**
```python
# At creation time:
val_loader = make_dataloader(val_subset, ..., split_type='validation')
val_loader._test_dataset_ref = test_dataset  # Track test set reference

# In tuning function:
if hasattr(val_loader, '_test_dataset_ref'):
    if val_loader.dataset is val_loader._test_dataset_ref:
        raise ValueError("Test set passed to hyperparameter tuning!")
```

---

## Summary of Findings

| Issue | Severity | Impact on Results | Fix Complexity |
|-------|----------|-------------------|----------------|
| #1: Augmentation Leakage | **CRITICAL** | All validation metrics invalid | Medium |
| #2: Resume Path Confusion | **CRITICAL** | Results lost, resume broken | Low |
| #3: Seed Isolation | **HIGH** | Multi-seed stats invalid | Low |
| #4: CSV Atomicity | **HIGH** | Data loss on crashes | Low |
| #5: Metric Naming | **MEDIUM** | Plotting/analysis broken | Low |
| #6: Gradient Accumulation | **MEDIUM** | LR scaling incorrect | Low |
| #7: Worker Seed | **MEDIUM** | Non-reproducible | Low |
| #8: Resume Inconsistency | **LOW** | Wasted compute | Low |
| #9: Test Set Leakage | **MEDIUM** | Potential overfitting | Medium |

---

## Recommended Fix Priority

1. **IMMEDIATE:** Fix augmentation leakage (#1) - invalidates all current validation results
2. **IMMEDIATE:** Fix resume path logic (#2) - prevents completing experiments
3. **HIGH:** Fix seed isolation (#3) - affects multi-seed reproducibility
4. **HIGH:** Fix CSV atomicity (#4) - prevents data loss
5. **MEDIUM:** Standardize metric naming (#5) - enables analysis
6. **MEDIUM:** Audit gradient accumulation (#6) - check if used anywhere
7. **MEDIUM:** Fix worker seeds (#7) - improves reproducibility
8. **LOW:** Standardize resume (#8) - improves UX
9. **MEDIUM:** Harden test set validation (#9) - prevents methodology errors

---

## Data Flow Map

```
Dataset Loading
  ├─ torchvision.datasets.MNIST/CIFAR10 (raw data)
  │
  ├─ Transform Application
  │   ├─ train_transform (WITH augmentation) ← BUG: Applied to val splits
  │   └─ eval_transform (NO augmentation)
  │
  ├─ Train/Val Split
  │   ├─ torch.randperm(..., generator=seeded_gen) ✓ Deterministic
  │   ├─ Subset(augmented_dataset, indices) ✗ INHERITS AUGMENTATION
  │   └─ Should use: TransformedSubset or separate datasets
  │
  ├─ DataLoader Creation
  │   ├─ make_dataloader(..., seed=X) ✓ Has worker_init_fn
  │   └─ DataLoader(...) direct calls ✗ Missing worker seeds
  │
  ├─ Training Loop
  │   ├─ for seed in seeds: ✓ Iterate seeds
  │   ├─   set_seed(seed) ✓ Set global seeds
  │   ├─   model = Model() ✓ New model
  │   ├─   optimizer = Opt(model.params) ✓ New optimizer
  │   ├─   try: train(...) ✓ Train
  │   └─   except: continue ✗ No cleanup → state leakage
  │
  ├─ Result Saving
  │   ├─ save_run_artifacts(results_dir, ..., history)
  │   ├─   df = pd.DataFrame(history) ✗ No metric normalization
  │   ├─   df.to_csv(path) ✗ Non-atomic write
  │   └─   meta.json with provenance ✓ Good metadata
  │
  └─ Resume Detection
      ├─ is_experiment_completed(results_dir, dataset, model, opt, seed)
      ├─   Check CSV exists ✓
      ├─   Check metadata ✓
      ├─   Path normalization ✗ Creates double-nesting bugs
      └─   Returns bool for skip decision
```

---

## Next Steps

1. Create TransformedSubset utility class
2. Implement atomic CSV writes
3. Fix all path handling to use canonical paths
4. Add seed isolation cleanup with try/finally
5. Enforce metric normalization at save time
6. Audit all DataLoader creation sites
7. Run validation suite to verify fixes

