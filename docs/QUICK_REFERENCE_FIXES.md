# GDSearch A* Publication Fixes - Quick Reference

## Audit Summary

**Original Verdict**: STRONG REJECT  
**Updated Status**: All critical flaws fixed, ready for re-evaluation  
**Date**: December 9, 2025

---

## 🚨 Critical Changes You MUST Make

### 1. Replace All Optimizer Wrappers (IMMEDIATE)

**❌ DO NOT USE** (Destroyed GPU acceleration):
```python
from src.core.pytorch_optimizers import AdamWrapper, SGDWrapper
```

**✅ USE INSTEAD**:
```python
from src.core.torch_native_optimizers import TorchAdam, TorchSGDMomentum, TorchAdamW, TorchSAM, TorchLookahead
# OR (Recommended):
from src.core.optimizer_registry import registry
optimizer = registry.create('Adam', model.parameters(), lr=0.001)
```

### 2. Enforce Data Hygiene (IMMEDIATE)

**❌ DO NOT DO THIS** (Data leakage):
```python
train_loader = DataLoader(train_set, ...)
test_loader = DataLoader(test_set, ...)
best_lr = tune_on_test(test_loader)  # WRONG! Leakage!
```

**✅ DO THIS INSTEAD**:
```python
from src.core.data_hygiene import DataSplitManager

manager = DataSplitManager(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Stage 1: Tune on val
train_loader = manager.get_train_loader()
val_loader = manager.get_val_loader()
best_hyperparams = tune(train_loader, val_loader)

# Freeze before test
manager.freeze_hyperparameters(best_hyperparams)

# Stage 2: Final eval on test
test_loader = manager.get_test_loader()
final_accuracy = evaluate(test_loader)
```

### 3. Use Correct Statistics (IMMEDIATE)

**❌ DO NOT USE** (Wrong for ranking):
```python
for i, j in pairs:
    t_stat, p = stats.ttest_ind(results[i], results[j])  # WRONG!
```

**✅ USE INSTEAD** (For N>2 optimizers):
```python
from src.analysis.statistical_analysis import friedman_test, nemenyi_test, plot_critical_difference_diagram

# data: shape (n_datasets, n_optimizers)
friedman = friedman_test(data, optimizer_names)
if friedman['significant']:
    nemenyi = nemenyi_test(data, optimizer_names)
    plot_critical_difference_diagram(
        nemenyi['mean_ranks'],
        optimizer_names,
        nemenyi['critical_distance'],
        save_path='results/cd_diagram.png'
    )
```

---

## 📊 Required Plots for Paper

Add these three plots to convince "Reviewer #2":

```python
from src.analysis.hessian_analysis import HessianAnalyzer
from src.analysis.statistical_analysis import plot_critical_difference_diagram

# 1. Critical Difference Diagram (Demšar 2006)
plot_critical_difference_diagram(mean_ranks, opt_names, cd)

# 2. Hessian Spectrum (Flatness proof)
analyzer = HessianAnalyzer(model, criterion)
results = analyzer.analyze_optimizer_quality(val_loader, 'SAM')
# → Shows SAM finds flatter minima than SGD

# 3. Sharpness Comparison (SAM validation)
sharpness = analyzer.compute_sharpness(val_loader, rho=0.05)
# → Lower sharpness = better generalization
```

---

## 🔧 Quick Migration

### Before Running Any Experiments:

1. **Update imports**:
   ```python
   # Add to top of experiment script:
   from src.core.torch_native_optimizers import *
   from src.core.data_hygiene import DataSplitManager
   from src.core.optimizer_registry import registry
   ```

2. **Replace hardcoded optimizer lists**:
   ```python
   # Create config file: configs/my_experiment.json
   {
     "optimizers": [
       {"name": "Adam", "lr": 0.001},
       {"name": "SGD_Momentum", "lr": 0.01, "momentum": 0.9}
     ]
   }
   
   # Load in script:
   from src.core.optimizer_registry import load_experiment_config
   opt_configs = load_experiment_config('configs/my_experiment.json')
   ```

3. **Add protocol validation**:
   ```python
   # At end of experiment:
   is_valid = manager.validate_protocol()
   if not is_valid:
       raise RuntimeError("Protocol violation detected!")
   ```

---

## 📝 Verification Checklist

Before paper submission:

- [ ] No imports from `src.core.pytorch_optimizers` (old wrappers)
- [ ] All experiments use `DataSplitManager` 
- [ ] Test set accessed ONLY after `freeze_hyperparameters()`
- [ ] Friedman + Nemenyi tests run (not just t-tests)
- [ ] Critical Difference diagram in paper
- [ ] Hessian spectrum analysis for SAM validation
- [ ] Sharpness comparison plot included
- [ ] `validate_protocol()` returns True

---

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `src/core/torch_native_optimizers.py` | ✅ GPU-optimized optimizers |
| `src/core/data_hygiene.py` | ✅ Prevent data leakage |
| `src/core/optimizer_registry.py` | ✅ Eliminate hardcoded lists |
| `src/analysis/statistical_analysis.py` | ✅ Friedman + Nemenyi |
| `src/analysis/hessian_analysis.py` | ✅ Flatness proof |
| `docs/AUDIT_FIXES_DECEMBER_2025.md` | 📄 Full documentation |

---

## ⚡ Quick Example: Full Pipeline

```python
from src.core.torch_native_optimizers import TorchAdam, TorchSAM
from src.core.data_hygiene import DataSplitManager
from src.analysis.statistical_analysis import friedman_test, nemenyi_test
from src.analysis.hessian_analysis import HessianAnalyzer

# 1. Data hygiene
manager = DataSplitManager(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

# 2. Hyperparameter tuning (Stage 1)
train_loader = manager.get_train_loader(batch_size=128)
val_loader = manager.get_val_loader(batch_size=256)

best_lr = tune_lr_on_validation(train_loader, val_loader)  # Val only!

# 3. Freeze hyperparameters
manager.freeze_hyperparameters({'lr': best_lr})

# 4. Final evaluation (Stage 2)
test_loader = manager.get_test_loader(batch_size=256)
model = create_model()
optimizer = TorchAdam(model.parameters(), lr=best_lr)

final_acc = train_and_evaluate(model, optimizer, train_loader, test_loader)

# 5. Statistical analysis (multi-seed)
# results shape: (n_seeds, n_optimizers)
friedman = friedman_test(results, ['Adam', 'SGD', 'SAM'])
nemenyi = nemenyi_test(results, ['Adam', 'SGD', 'SAM'])

# 6. Hessian analysis (flatness proof)
analyzer = HessianAnalyzer(model, nn.CrossEntropyLoss())
hessian_results = analyzer.analyze_optimizer_quality(val_loader, 'SAM')

# 7. Validate protocol
assert manager.validate_protocol(), "Protocol violation!"
```

---

**Status**: All critical fixes implemented ✅  
**Next**: Re-run experiments with corrected code → Submit to NeurIPS/ICML
