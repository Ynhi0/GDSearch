# EXPERIMENT ISOLATION FIXES - Action Plan

## Quick Summary

**Status:** ✅ Overall system is production-ready  
**Critical Issues:** 0  
**High Priority:** 0  
**Medium Priority:** 2 (should fix in next sprint)  
**Low Priority:** 3 (nice to have)

---

## MEDIUM PRIORITY FIXES (2-4 hours total)

### Fix #1: Eliminate Global Flag Coupling

**Problem:** 12 global flags (`USE_AMP`, `USE_EMA`, etc.) are set in `main()` and implicitly read by experiments.

**File:** `run_all_kaggle.py`

**Current (Lines 1856-1875):**
```python
# Global flags for advanced training features
USE_AMP = False
USE_EMA = False
LABEL_SMOOTHING = 0.0
GRADIENT_CLIP_NORM = None
USE_AGC = False
USE_ROBUST_LOSS = False
USE_TRIMMED_MEAN = False
MONITOR_HEAVY_TAILS = True
ENABLE_LOSS_LANDSCAPE = False

# Set in main():
USE_AMP = args.use_amp or args.kaggle_t4
USE_EMA = args.use_ema
# ... (experiments read these globals implicitly)
```

**Solution:** Create config dataclass and pass explicitly

```python
# Add to top of file:
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration passed explicitly to experiments."""
    use_amp: bool = False
    use_ema: bool = False
    label_smoothing: float = 0.0
    gradient_clip_norm: Optional[float] = None
    use_agc: bool = False
    use_robust_loss: bool = False
    use_trimmed_mean: bool = False
    monitor_heavy_tails: bool = True
    enable_loss_landscape: bool = False
    auto_lr_enabled: bool = False
    adaptive_batch_enabled: bool = False

# In main(), create config:
training_config = TrainingConfig(
    use_amp=args.use_amp or args.kaggle_t4,
    use_ema=args.use_ema,
    label_smoothing=args.label_smoothing,
    gradient_clip_norm=args.gradient_clip_norm,
    # ... etc
)

# Pass to experiments:
experiment_results['mnist'] = run_mnist_experiment(
    results_dir=str(experiments_dir / "mnist"),
    seeds=seeds,
    training_config=training_config,  # ← NEW
    tracker=tracker,
    # ...
)

# Update function signatures:
def run_mnist_experiment(
    results_dir="results_mnist",
    seeds=None,
    training_config: TrainingConfig = None,  # ← NEW
    tracker=None,
    # ...
):
    # Use config instead of globals:
    if training_config and training_config.use_amp:
        # ...
```

**Benefits:**
- ✅ Experiments can be tested in isolation
- ✅ No hidden dependencies on global state
- ✅ Explicit configuration in function signatures
- ✅ Type-safe with dataclass

**Estimated Effort:** 2-3 hours

---

### Fix #2: Create Fresh Tracker/Profiler Per Experiment

**Problem:** All experiments share same `tracker` and `profiler` instances, creating implicit coupling.

**File:** `run_all_kaggle.py` (Lines 10150-10200)

**Current:**
```python
# Created once in main():
tracker = ExperimentTracker()
profiler = PerformanceProfiler() if args.profile else None

# Shared across all experiments:
experiment_results['mnist'] = run_mnist_experiment(..., tracker=tracker, profiler=profiler)
experiment_results['cifar10'] = run_cifar10_experiment(..., tracker=tracker, profiler=profiler)
```

**Solution:** Create fresh instances per experiment

```python
# Helper function to create experiment context:
def create_experiment_context(args):
    """Create fresh tracker and profiler instances."""
    tracker = None
    if not args.no_mlflow:
        try:
            tracker = ExperimentTracker()
        except Exception as e:
            logging.warning(f"Failed to create tracker: {e}")
    
    profiler = PerformanceProfiler() if args.profile else None
    
    return tracker, profiler

# In experiment loop:
for exp_name in selected_experiments:
    # Create fresh instances
    tracker, profiler = create_experiment_context(args)
    
    if exp_name == 'mnist':
        experiment_results['mnist'] = run_mnist_experiment(
            results_dir=str(experiments_dir / "mnist"),
            seeds=seeds,
            tracker=tracker,  # ← Fresh instance
            profiler=profiler,  # ← Fresh instance
            # ...
        )
    
    # Clean up
    if tracker:
        tracker.end_run()
    
    clear_gpu_memory(force=True)
```

**Benefits:**
- ✅ Complete isolation between experiments
- ✅ No state leakage through shared instances
- ✅ Easier to parallelize in future

**Estimated Effort:** 1-2 hours

---

## LOW PRIORITY FIXES (1 hour total)

### Fix #3: Add GPU Cleanup to Experiment Functions

**Problem:** Some experiment runners don't explicitly clear GPU memory at end.

**Files:** 
- `src/experiments/run_cifar10.py`
- `src/experiments/run_nn_experiment.py`
- `src/experiments/run_transformer_nlp.py`

**Solution:** Add cleanup at end of each function

```python
def run_single(optimizer_name: str, seed: int, ...):
    # ... existing code ...
    
    df.to_csv(out, index=False)
    
    # NEW: Cleanup GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return out
```

**Files to Update:**
1. `src/experiments/run_cifar10.py:165-220`
2. `src/experiments/run_nn_experiment.py:240-640`
3. `src/experiments/run_transformer_nlp.py:139-420`
4. `src/experiments/run_medical_segmentation.py:80-155`

**Estimated Effort:** 15 minutes

---

### Fix #4: Add Nested MLflow Runs Per Seed

**Problem:** All seeds share parent MLflow run, making organization less clean.

**Current:**
```python
tracker.start_run(run_name="MNIST_Run")
for seed in seeds:
    for opt_name in optimizers:
        # No nested run - all logged to parent
        train_model(...)
tracker.end_run()
```

**Solution:**
```python
tracker.start_run(run_name="MNIST_Run")
for seed in seeds:
    for opt_name in optimizers:
        # Create child run per experiment
        tracker.start_run(run_name=f"{opt_name}_seed{seed}")
        train_model(...)
        tracker.end_run()
tracker.end_run()
```

**Files to Update:**
- `run_all_kaggle.py`: MNIST experiment (Line 3245-3710)
- `run_all_kaggle.py`: CIFAR-10 experiment (Line 3888-4400)

**Estimated Effort:** 30 minutes

---

### Fix #5: Remove Global Config Dictionary

**Problem:** `globals()['EXPERIMENT_CONFIG'] = experiment_config` creates global state.

**Current (Line 9756-9758):**
```python
if experiment_config:
    globals()['EXPERIMENT_CONFIG'] = experiment_config
```

**Solution:** Pass as parameter (already handled by Fix #1)

**Estimated Effort:** Included in Fix #1

---

## TESTING PLAN

### 1. Add Test: Experiment Order Independence

**File:** `tests/test_experiment_isolation.py` (NEW)

```python
"""Test experiment isolation and order independence."""
import pytest
from pathlib import Path
from run_all_kaggle import run_mnist_experiment, run_cifar10_experiment, clear_gpu_memory

def test_experiment_order_independence(tmp_path):
    """Verify experiments produce identical results regardless of execution order."""
    
    # Order A -> B
    results_ab = {}
    results_ab['mnist'] = run_mnist_experiment(
        results_dir=str(tmp_path / "order_ab" / "mnist"),
        seeds=[42],
        quick=True,
        skip_tuning=True
    )
    clear_gpu_memory(force=True)
    
    results_ab['cifar10'] = run_cifar10_experiment(
        results_dir=str(tmp_path / "order_ab" / "cifar10"),
        seeds=[42],
        quick=True,
        skip_tuning=True
    )
    clear_gpu_memory(force=True)
    
    # Order B -> A
    results_ba = {}
    results_ba['cifar10'] = run_cifar10_experiment(
        results_dir=str(tmp_path / "order_ba" / "cifar10"),
        seeds=[42],
        quick=True,
        skip_tuning=True
    )
    clear_gpu_memory(force=True)
    
    results_ba['mnist'] = run_mnist_experiment(
        results_dir=str(tmp_path / "order_ba" / "mnist"),
        seeds=[42],
        quick=True,
        skip_tuning=True
    )
    
    # Compare final metrics (within tolerance for CUDA non-determinism)
    mnist_ab = results_ab['mnist']
    mnist_ba = results_ba['mnist']
    
    # Both should have same structure
    assert len(mnist_ab) == len(mnist_ba)
    
    # Final test accuracy should be within 1% (allowing CUDA variance)
    final_acc_ab = mnist_ab['test_accuracy'].iloc[-1]
    final_acc_ba = mnist_ba['test_accuracy'].iloc[-1]
    assert abs(final_acc_ab - final_acc_ba) < 0.01
```

### 2. Add Test: Global State Isolation

```python
def test_global_state_not_mutated():
    """Verify experiments don't pollute global state."""
    
    # Import globals to check
    import run_all_kaggle as rak
    
    # Record initial state
    initial_use_amp = rak.USE_AMP
    initial_use_ema = rak.USE_EMA
    
    # Run experiment
    run_mnist_experiment(
        results_dir="tmp/test_global_state",
        seeds=[42],
        quick=True,
        skip_tuning=True
    )
    
    # Verify globals unchanged
    assert rak.USE_AMP == initial_use_amp
    assert rak.USE_EMA == initial_use_ema
```

### 3. Add Test: MLflow Run Isolation

```python
def test_mlflow_run_isolation():
    """Verify MLflow runs are properly isolated."""
    
    tracker = ExperimentTracker()
    
    # Start parent run
    parent_id = tracker.start_run(run_name="Parent")
    assert tracker.current_run is not None
    
    # Start nested run
    child_id = tracker.start_run(run_name="Child")
    assert child_id != parent_id
    assert len(tracker.run_stack) == 1
    
    # End nested run
    tracker.end_run()
    assert tracker.current_run.info.run_id == parent_id
    
    # End parent run
    tracker.end_run()
    assert tracker.current_run is None
    assert len(tracker.run_stack) == 0
```

---

## ROLLOUT PLAN

### Phase 1: Low Priority Fixes (Week 1)
- ✅ Add GPU cleanup to experiment functions (Fix #3)
- ✅ Add nested MLflow runs (Fix #4)
- ✅ Write isolation tests

**Risk:** Low - These are additive changes

---

### Phase 2: Medium Priority Fixes (Week 2)
- ⚠️ Create TrainingConfig dataclass (Fix #1)
- ⚠️ Pass config explicitly to all experiments
- ⚠️ Create fresh tracker/profiler per experiment (Fix #2)
- ✅ Update all tests

**Risk:** Medium - Changes function signatures

**Migration Strategy:**
1. Add TrainingConfig with default values
2. Make parameter optional initially: `training_config: TrainingConfig = None`
3. Fall back to globals if None: `config = training_config or TrainingConfig(use_amp=USE_AMP, ...)`
4. Update all callers
5. Remove fallback and make parameter required

---

### Phase 3: Validation (Week 3)
- Run full benchmark suite with fixes
- Compare results with baseline (should be identical)
- Update documentation
- Deploy to production

---

## VERIFICATION CHECKLIST

After implementing fixes, verify:

- [ ] All tests pass: `pytest tests/test_experiment_isolation.py -v`
- [ ] Experiments can run in any order: `test_experiment_order_independence`
- [ ] Global state unchanged after experiments: `test_global_state_not_mutated`
- [ ] MLflow runs properly nested: `test_mlflow_run_isolation`
- [ ] GPU memory properly cleared: Monitor with `nvidia-smi`
- [ ] Results identical to baseline: Compare CSV outputs
- [ ] No filename collisions: Check `results/experiments/` structure
- [ ] Checkpoints resume correctly: Test with `--resume` flag

---

## CONTACT

For questions or issues during implementation:
- Review full audit: `EXPERIMENT_ISOLATION_AUDIT.md`
- Check existing tests: `tests/test_*.py`
- Consult copilot instructions: `.github/copilot-instructions.md`
