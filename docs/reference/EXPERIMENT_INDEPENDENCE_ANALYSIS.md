# Experiment Independence Analysis for GDSearch

**Analysis Date:** February 2, 2026  
**Repository:** c:\Users\MPhuc\Desktop\GDSearch  
**Main Orchestrator:** [run_all_kaggle.py](run_all_kaggle.py)

---

## Executive Summary

**ANSWER: YES - Experiments are FULLY INDEPENDENT**

✅ **Key Findings:**
1. **Experiments can run independently** - No hard dependencies between experiments
2. **Experiments can run in any order** - No sequential requirements
3. **Experiments can run in parallel** - No shared state or file conflicts
4. **Single seed can run independently** - Each seed is completely isolated
5. **Resume mode is safe** - Only checks for existing files, doesn't create dependencies

---

## Detailed Analysis

### 1. Experiment Orchestration Model

**Location:** [run_all_kaggle.py](run_all_kaggle.py) lines 9400-10400

#### Execution Pattern
```python
# Sequential execution with independent experiments
if 'mnist' in selected_experiments:
    experiment_results['mnist'] = run_mnist_experiment(...)
    
if 'cifar10' in selected_experiments:
    experiment_results['cifar10'] = run_cifar10_experiment(...)
    
if 'nlp' in selected_experiments:
    experiment_results['nlp'] = run_nlp_experiment(...)
```

**Pattern:** Each experiment is wrapped in an independent `if` block. They execute sequentially but have **NO data dependencies**.

#### Evidence from Code:
- Line 9641: `experiment_results['mnist'] = run_mnist_experiment(...)`
- Line 9659: `experiment_results['cifar10'] = run_cifar10_experiment(...)`
- Line 9679: `experiment_results['nlp'] = run_nlp_experiment(...)`

Each experiment:
- Gets its own `results_dir` subdirectory
- Uses only its own dataset
- Saves to unique file paths
- Does NOT read from other experiments

---

### 2. Data Flow Architecture

#### Input Data (Per Experiment)
Each experiment loads its own data independently:

**MNIST Experiment** ([run_all_kaggle.py](run_all_kaggle.py) line 2727):
```python
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True)
```

**CIFAR10 Experiment** ([run_all_kaggle.py](run_all_kaggle.py) line 3590):
```python
train_ds = torchvision.datasets.CIFAR10('./data', train=True, download=True)
test_ds = torchvision.datasets.CIFAR10('./data', train=False, download=True)
```

**No Shared Loading:** Each dataset is loaded independently from `./data/` directory.

#### Output Data (Per Experiment)
Each experiment writes to **isolated directories**:

| Experiment | Output Directory | File Pattern |
|------------|------------------|--------------|
| MNIST | `results/experiments/mnist/` | `MNIST_SimpleMLP_{optimizer}_seed{N}.csv` |
| CIFAR10 | `results/experiments/cifar10/` | `CIFAR10_ResNet18_{optimizer}_seed{N}.csv` |
| NLP | `results/experiments/nlp/` | `NLP_{model}_{optimizer}_seed{N}.csv` |
| Medical | `results/experiments/medical/` | `Medical_UNet_{optimizer}_seed{N}.csv` |
| 2D Optimization | `results/experiments/2d_optimization/` | `2D_{function}_{optimizer}_seed{N}.csv` |

**File Naming Convention:** ([run_all_kaggle.py](run_all_kaggle.py) line 1048)
```python
file_stem = f"{dataset}_{model_name}_{optimizer_name}_seed{seed}"
csv_path = results_base / f"{file_stem}.csv"
```

#### Cross-Experiment Data Flow: NONE

**Evidence:** Searched for cross-experiment reads:
```bash
grep -r "load.*results.*from" src/experiments/*.py
grep -r "read.*csv.*mnist" src/experiments/*.py
```

**Result:** Only ONE module reads other experiments' results:
- `theory_practice_validation.py` (lines 81-147) - Analysis module (optional)
- `generate_final_deliverables.py` - Report generator (optional)

**Critical Finding:** These are **post-hoc analysis modules**, NOT part of the core experiment pipeline. They only run if explicitly enabled with `--with-theory-analysis` or `--generate-deliverables`.

---

### 3. Resume Logic Analysis

**Location:** [src/core/resume_utils.py](src/core/resume_utils.py)

#### Resume Detection Mechanism

**Function:** `is_experiment_completed()` ([run_all_kaggle.py](run_all_kaggle.py) line 1020)

```python
def is_experiment_completed(results_dir, dataset, model_name, optimizer_name, seed):
    """Check if experiment completed by looking for result files"""
    csv_path = results_base / f"{dataset}_{model_name}_{optimizer_name}_seed{seed}.csv"
    meta_path = results_base / f"{dataset}_{model_name}_{optimizer_name}_seed{seed}.metadata.json"
    
    # Check if files exist and have data
    if meta_path.exists():
        meta = json.load(open(meta_path))
        if meta.get('completed', False) or meta.get('rows', 0) >= 1:
            return True
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) >= 1:
            return True
    
    return False
```

#### Resume Behavior Modes

**CLI Flag:** `--resume-behavior` (choices: `error_if_no_checkpoint`, `restart_if_no_checkpoint`, `skip_if_results_exist`)

**Default:** `skip_if_results_exist` when `--resume` is set

**Logic:** ([src/core/resume_utils.py](src/core/resume_utils.py) line 118)
```python
def decide_resume_action(checkpoint, results_dir, signature, resume_behavior):
    if checkpoint is not None:
        if checkpoint.get('metadata', {}).get('completed', False):
            return 'skip'  # Already done
        return 'restart'  # Resume from checkpoint
    
    # No checkpoint:
    if resume_behavior == 'skip_if_results_exist':
        if results_exist(results_dir, signature):
            return 'skip'  # Skip if results found
        return 'restart'  # Start fresh
```

#### Independence Guarantee

**Critical:** Resume logic **ONLY** checks:
1. Existence of result files for **SAME** (dataset, model, optimizer, seed)
2. Existence of checkpoint files for **SAME** configuration

**It does NOT:**
- ❌ Check if other experiments completed
- ❌ Read other experiments' results
- ❌ Create dependencies between experiments

**Evidence:** Line 2945, 3642, 4185 - All skip checks are self-contained:
```python
if resume and is_experiment_completed(results_dir, 'MNIST', 'SimpleMLP', opt_name, seed):
    logging.info(f"Skipping {opt_name} seed {seed} (already completed)")
    continue
```

---

### 4. Checkpoint System

**Location:** [src/core/checkpoint_manager.py](src/core/checkpoint_manager.py)

#### Checkpoint Isolation

**Per-Run Checkpoints:**
```python
# Line 3023: MNIST checkpoint
ckpt_file = f"MNIST_{opt_name}_seed{seed}.pt"
checkpoint = checkpoint_manager.load_checkpoint(ckpt_file, f"MNIST_{opt_name}_seed{seed}")

# Line 3689: CIFAR-10 checkpoint
ckpt_file = f"CIFAR10_{opt_name}_seed{seed}.pt"
checkpoint = checkpoint_manager.load_checkpoint(ckpt_file, f"CIFAR10_{opt_name}_seed{seed}")
```

**Checkpoint Contents:** ([run_all_kaggle.py](run_all_kaggle.py) line 3341)
```python
checkpoint_data = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
    'rng_states': {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_cpu_rng_state': torch.get_rng_state(),
        'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all()
    },
    'metadata': {
        'completed': (epoch >= epochs),
        'config': config
    }
}
```

**Isolation Guarantee:**
- Each checkpoint is specific to (experiment, optimizer, seed)
- No shared checkpoint state
- No cross-experiment checkpoint dependencies

---

### 5. Shared State Analysis

#### Global State: MINIMAL

**Search Results:**
```bash
grep -r "global " run_all_kaggle.py
```

**Found:**
- `global_step` - Local training loop counter (line 321 in train loop)
- `ULTRA_QUICK_MODE` - Read-only config flag

**No shared mutable global state** that creates dependencies.

#### Configuration System

**Hyperparameter Source:** ([run_all_kaggle.py](run_all_kaggle.py) line 1097)
```python
def load_experiment_config(config_path=None):
    # Loads from configs/benchmark_hyperparameters.json
    # Each experiment gets its own section
    return default_config
```

**Config Structure:** ([configs/benchmark_hyperparameters.json](configs/benchmark_hyperparameters.json))
```json
{
  "experiment_configs": {
    "2d_optimization": { "optimizers": {...} },
    "resnet_cifar10": { "optimizers": {...} },
    "highdim_optimization": { "optimizers": {...} }
  }
}
```

**Independence:** Each experiment reads its own config section. No cross-references.

#### Hyperparameter Tuning

**Location:** [run_all_kaggle.py](run_all_kaggle.py) line 2753 (MNIST), 3710 (CIFAR10)

**Pattern:**
```python
if not skip_tuning:
    # Tune hyperparameters for THIS experiment only
    tuned_params = {}
    for opt_name in optimizers_to_tune:
        # Use Optuna on validation split of THIS dataset
        tuned_params[opt_name] = tune_hyperparameters(...)
```

**Critical:** Tuning happens **per-experiment**, NOT globally:
- MNIST tuning uses MNIST validation set
- CIFAR10 tuning uses CIFAR10 validation set
- Results stored in `tuned_params` local variable
- NOT saved to disk for other experiments

**Evidence:** No cross-experiment tuning:
- Line 2753: MNIST tuning is self-contained
- Line 3710: CIFAR10 tuning starts fresh
- No config files are read/written during tuning

---

### 6. Parallel Execution Safety

#### Current Implementation: Sequential

**Code Pattern:**
```python
if 'mnist' in selected_experiments:
    experiment_results['mnist'] = run_mnist_experiment(...)

if 'cifar10' in selected_experiments:
    experiment_results['cifar10'] = run_cifar10_experiment(...)
```

**Sequential by design**, but architecture supports parallelization.

#### Parallel Safety Analysis

**Can experiments run in parallel?** ✅ **YES**

**Evidence:**

1. **No shared files:** Each experiment writes to unique paths
   - MNIST: `results/experiments/mnist/*.csv`
   - CIFAR-10: `results/experiments/cifar10/*.csv`

2. **No process synchronization:** No locks, semaphores, or shared memory

3. **No subprocess forking:** Search for `subprocess.run` shows:
   - Line 1182: Git hash (metadata only)
   - Line 1220: nvidia-smi (metadata only)
   - Line 6426: Plot generation (post-hoc)

4. **Dataset loading is thread-safe:** PyTorch DataLoader is multiprocessing-safe

5. **GPU isolation:** Each experiment can use different GPU if available:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

#### Potential Conflicts (if parallelized)

| Resource | Conflict Risk | Mitigation |
|----------|---------------|------------|
| **GPU Memory** | ⚠️ HIGH | Assign different GPUs or use batch size limiting |
| **Disk Space** | ⚠️ MEDIUM | Monitor with `DiskSpaceGuardian` (already implemented) |
| **Data Download** | ⚠️ LOW | torchvision handles concurrent downloads with locks |
| **File Writes** | ✅ NONE | Unique file paths per experiment |
| **Checkpoints** | ✅ NONE | Unique checkpoint files per (exp, opt, seed) |
| **MLflow** | ⚠️ LOW | MLflow handles concurrent writes |

**Conclusion:** Parallelization is **SAFE** with GPU memory management.

---

### 7. Experiment Dependency Matrix

```
           MNIST  CIFAR10  NLP  Medical  2D  Robustness  Ablation
MNIST        -      ✅      ✅     ✅     ✅      ✅         ✅
CIFAR10      ✅      -      ✅     ✅     ✅      ✅         ✅
NLP          ✅      ✅      -     ✅     ✅      ✅         ✅
Medical      ✅      ✅     ✅      -     ✅      ✅         ✅
2D           ✅      ✅     ✅     ✅      -      ✅         ✅
Robustness   ✅      ✅     ✅     ✅     ✅       -         ✅
Ablation     ✅      ✅     ✅     ✅     ✅      ✅          -

✅ = Can run independently (no dependency)
```

**All entries are ✅ - Full independence confirmed.**

---

### 8. Post-Hoc Analysis Modules

**Two modules read experiment results AFTER completion:**

#### Theory-Practice Validation

**File:** [src/experiments/theory_practice_validation.py](src/experiments/theory_practice_validation.py)  
**Trigger:** `--with-theory-analysis` flag  
**Behavior:**
```python
def load_training_results(results_dir, experiment='mnist'):
    experiment_dir = Path(results_dir) / experiment
    csv_files = list(experiment_dir.glob("*.csv"))
    # Reads completed CSVs for analysis
```

**Dependency:** Reads experiment results BUT:
- Only runs if explicitly enabled
- Runs AFTER all experiments complete
- Does NOT affect experiment execution
- Is purely analytical

#### Final Deliverables Generator

**File:** [src/experiments/generate_final_deliverables.py](src/experiments/generate_final_deliverables.py)  
**Trigger:** `--generate-deliverables` flag  
**Behavior:**
```python
def generate_all():
    csv_files = list(self.results_dir.glob("**/NN_*.csv"))
    # Aggregates results for reports
```

**Dependency:** Same as theory validation - post-hoc only.

---

## Answers to Specific Questions

### 1. Can I run MNIST experiment without running CIFAR10?

**✅ YES**

**Command:**
```bash
python run_all_kaggle.py --experiments mnist --quick --seeds 42
```

**Evidence:**
- Line 9641: MNIST runs in independent `if 'mnist' in selected_experiments` block
- No code reads CIFAR10 results in MNIST experiment
- Output goes to `results/experiments/mnist/`

---

### 2. Can I run experiments in any order?

**✅ YES**

**Examples:**
```bash
# Order 1: MNIST → CIFAR10 → NLP
python run_all_kaggle.py --experiments mnist,cifar10,nlp

# Order 2: NLP → CIFAR-10 → MNIST (reversed)
python run_all_kaggle.py --experiments nlp,cifar10,mnist

# Order 3: Random selection
python run_all_kaggle.py --experiments 2d,medical,ablation
```

**Evidence:**
- No sequential dependencies in code
- Each experiment reads only its own data
- Resume logic only checks same-experiment completion

---

### 3. Can I run experiments in parallel?

**✅ YES (with GPU memory management)**

**Safe Parallel Execution:**

**Option 1: Multiple Processes**
```bash
# Terminal 1
python run_all_kaggle.py --experiments mnist --quick &

# Terminal 2  
python run_all_kaggle.py --experiments cifar10 --quick &

# Terminal 3
python run_all_kaggle.py --experiments nlp --quick &
```

**Option 2: Assign Different GPUs**
```bash
CUDA_VISIBLE_DEVICES=0 python run_all_kaggle.py --experiments mnist --quick &
CUDA_VISIBLE_DEVICES=1 python run_all_kaggle.py --experiments cifar10 --quick &
```

**Conflicts:**
- ⚠️ **GPU Memory:** Multiple experiments on same GPU may cause OOM
- ✅ **File System:** No conflicts (unique file paths)
- ✅ **Data Loading:** Thread-safe
- ✅ **Checkpoints:** Isolated per experiment

**Recommendation:** Use `--kaggle-t4` for optimized batch sizes or assign different GPUs.

---

### 4. Can I run single seed without other seeds?

**✅ YES**

**Command:**
```bash
python run_all_kaggle.py --experiments mnist --seeds 42 --quick
```

**Evidence:**
- Line 2943: Loop over seeds with independent iterations:
  ```python
  for seed in seeds:
      set_seed(seed)
      # Each seed iteration is isolated
  ```
- Each seed creates unique files: `*_seed42.csv`, `*_seed123.csv`
- No aggregation during experiment (only in post-analysis)

---

### 5. Can I skip experiment A and still run experiment B?

**✅ YES for ALL pairs**

**Examples:**

```bash
# Skip MNIST, run only CIFAR10
python run_all_kaggle.py --experiments cifar10

# Skip MNIST and CIFAR10, run only NLP
python run_all_kaggle.py --experiments nlp

# Run only ablation studies
python run_all_kaggle.py --experiments ablation,batch_ablation,lr_ablation
```

**Evidence:**
- Each experiment checks `if 'name' in selected_experiments`
- No cross-experiment dependencies
- Experiments can run in isolation

**Caveat:** Post-hoc analysis (`--with-theory-analysis`) expects experiments to exist. If running analysis, must have prior experiment results.

---

### 6. Does resume mode create dependencies?

**❌ NO**

**Resume Logic:**
```python
if resume and is_experiment_completed(results_dir, 'MNIST', 'SimpleMLP', opt_name, seed):
    logging.info(f"Skipping {opt_name} seed {seed} (already completed)")
    continue
```

**What resume checks:**
- ✅ Does THIS exact (experiment, optimizer, seed) have results?
- ❌ Does NOT check other experiments
- ❌ Does NOT read other experiments' data

**Resume modes:**
- `skip_if_results_exist` (default): Skip if CSV exists for THIS run
- `restart_if_no_checkpoint`: Restart THIS run if no checkpoint
- `error_if_no_checkpoint`: Error if THIS run has no checkpoint

**All modes are self-contained per experiment.**

---

### 7. Do experiments share checkpoints or models?

**❌ NO**

**Checkpoint Isolation:**
```python
# MNIST checkpoint
ckpt_file = f"MNIST_{opt_name}_seed{seed}.pt"

# CIFAR10 checkpoint  
ckpt_file = f"CIFAR10_{opt_name}_seed{seed}.pt"

# NLP checkpoint
ckpt_file = f"IMDB_{model_name}_{opt_name}_seed{seed}.pt"
```

**Evidence:**
- Each experiment uses unique checkpoint filenames
- No checkpoint sharing code
- No transfer learning between experiments

**Model Architectures:**
- MNIST: `SimpleMLP` (784 → 128 → 64 → 10)
- CIFAR10: `ResNet18` (3×32×32 → 10 classes)
- NLP: Transformer models (task-specific)

**No weight sharing or model reuse.**

---

### 8. Does hyperparameter tuning affect other experiments?

**❌ NO**

**Tuning Scope:**
```python
# MNIST tuning (line 2753)
if not skip_tuning:
    tuned_params = {}  # LOCAL variable
    for opt_name in optimizers_to_tune:
        # Tune on MNIST validation set
        tuned_params[opt_name] = tune_hyperparameters(...)
    # tuned_params used only for MNIST experiment
```

**Evidence:**
- Tuning results stored in **local variables**
- NOT saved to disk
- NOT read by other experiments
- Each experiment tunes independently

**Fair Comparison Note:** ([run_all_kaggle.py](run_all_kaggle.py) line 2790)
```python
# All optimizers receive equal tuning budget
n_trials = 5 if quick else 15
tune_epochs = 1 if ULTRA_QUICK_MODE else 3
```

**All optimizers get same tuning budget per experiment**, but results are local.

---

## Execution Models

### Current: Sequential Execution

```
Start → Load Config → Select Experiments
  ↓
  ├─→ MNIST (if selected) → Save to results/experiments/mnist/
  ↓
  ├─→ CIFAR10 (if selected) → Save to results/experiments/cifar10/
  ↓
  ├─→ NLP (if selected) → Save to results/experiments/nlp/
  ↓
  └─→ Generate Analysis (optional) → Aggregate results
```

### Proposed: Parallel Execution (Safe)

```
                    ┌─→ MNIST (GPU 0) → results/experiments/mnist/
Start → Load Config ├─→ CIFAR10 (GPU 1) → results/experiments/cifar10/
                    └─→ NLP (CPU) → results/experiments/nlp/
                           ↓
                    Wait for all → Generate Analysis
```

---

## Practical Recommendations

### For Independent Experiment Runs

**Run single experiment:**
```bash
python run_all_kaggle.py --experiments mnist --quick --seeds 42,123,456
```

**Run multiple experiments (any order):**
```bash
python run_all_kaggle.py --experiments nlp,2d,robustness --quick
```

**Run single optimizer, single seed:**
```bash
# Edit run_all_kaggle.py line 2853 to filter optimizers
python run_all_kaggle.py --experiments mnist --seeds 42 --quick
```

### For Parallel Execution

**Safe approach (different GPUs):**
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python run_all_kaggle.py --experiments mnist --quick &

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python run_all_kaggle.py --experiments cifar10 --quick &
```

**Safe approach (CPU for some):**
```bash
# Terminal 1 (GPU)
python run_all_kaggle.py --experiments mnist,cifar10 --quick &

# Terminal 2 (CPU)
CUDA_VISIBLE_DEVICES="" python run_all_kaggle.py --experiments 2d --quick &
```

### For Resume/Incremental Runs

**Run new experiments, skip completed:**
```bash
python run_all_kaggle.py --experiments mnist,cifar10,nlp --resume --quick
```

**Restart specific experiment:**
```bash
# Delete results for MNIST
rm -rf results/experiments/mnist/

# Rerun only MNIST
python run_all_kaggle.py --experiments mnist --quick
```

---

## Conclusion

### Summary Table

| Question | Answer | Evidence Location |
|----------|--------|-------------------|
| **Independent execution?** | ✅ YES | [run_all_kaggle.py](run_all_kaggle.py) L9641-9734 |
| **Any order?** | ✅ YES | No sequential dependencies found |
| **Parallel execution?** | ✅ YES* | Unique file paths, isolated state (*GPU mem) |
| **Single seed?** | ✅ YES | [run_all_kaggle.py](run_all_kaggle.py) L2943 loop |
| **Skip experiments?** | ✅ YES | `--experiments` flag filters |
| **Resume creates deps?** | ❌ NO | [src/core/resume_utils.py](src/core/resume_utils.py) L118 |
| **Shared checkpoints?** | ❌ NO | Unique filenames per (exp, opt, seed) |
| **Tuning affects others?** | ❌ NO | Local variables, not saved |

### Key Architectural Strengths

1. **Clean Separation:** Each experiment is self-contained
2. **Unique File Paths:** No file conflicts
3. **Local State:** No shared global state
4. **Independent Data:** Each loads its own dataset
5. **Per-Run Checkpoints:** Isolated checkpoint files
6. **Resume Safety:** Only checks own completion

### Parallelization Roadmap

If you want to parallelize experiments in the future:

**Minimal Changes Required:**
```python
# Instead of:
if 'mnist' in selected_experiments:
    experiment_results['mnist'] = run_mnist_experiment(...)

# Use:
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=3) as executor:
    futures = {}
    if 'mnist' in selected_experiments:
        futures['mnist'] = executor.submit(run_mnist_experiment, ...)
    if 'cifar10' in selected_experiments:
        futures['cifar10'] = executor.submit(run_cifar10_experiment, ...)
    
    # Wait for completion
    for name, future in futures.items():
        experiment_results[name] = future.result()
```

**Only consideration:** GPU memory limits.

---

## References

**Key Files Analyzed:**
- [run_all_kaggle.py](run_all_kaggle.py) - Main orchestrator (10,816 lines)
- [src/core/resume_utils.py](src/core/resume_utils.py) - Resume logic
- [src/core/checkpoint_manager.py](src/core/checkpoint_manager.py) - Checkpoint system
- [src/experiments/run_nn_experiment.py](src/experiments/run_nn_experiment.py) - Core training loop
- [configs/benchmark_hyperparameters.json](configs/benchmark_hyperparameters.json) - Configuration

**Analysis Method:**
- Code review with line-by-line trace
- Data flow mapping
- Dependency graph construction
- File I/O analysis
- Global state inspection

---

**Report Generated:** February 2, 2026  
**Status:** ✅ COMPLETE - All questions answered with code evidence
