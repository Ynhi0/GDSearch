# REMAINING TODO - Quick Action List

**Last Updated**: December 9, 2025  
**Status**: 8/12 items complete (67%)

---

## ✅ COMPLETED (No Action Needed)

- [x] Fix data leakage (validation split implemented)
- [x] Fix broken code (optuna script runs)
- [x] Pin dependencies (requirements.txt updated)
- [x] Baseline fairness audit (10 tests passing)
- [x] Config validation tests (automated)
- [x] Auto-wiring safety audit (no issues found)
- [x] Hardware agnosticism check (all good)
- [x] Zombie config detection (script created)

---

## 🔴 HIGH PRIORITY - Required for Publication

### 1. Re-run All Experiments (3-5 days) ⚠️
**Why**: Previous results may contain test set leakage contamination

**Steps**:
```bash
# Backup old results
mv results results_pre_audit_backup
mkdir results

# Run multi-seed experiments with validation split
python src/experiments/run_multi_seed.py \
  --config configs/nn_tuning.json \
  --seeds 42,123,456,789,101112

python src/experiments/run_multi_seed.py \
  --config configs/cifar10_tuning.json \
  --seeds 42,123,456,789,101112

# Kaggle GPU benchmarks
cd kaggle
python run_all_kaggle.py --experiments mnist,cifar10,nlp --seeds 42,123,456

# Regenerate visualizations
python src/visualization/plot_results.py --results-dir ../results
```

**Deliverables**:
- New results/ directory with clean data
- Updated figures for paper
- Recalculated statistics

---

### 2. Model Architecture Standardization (2 days) ⚠️
**Problem**: `run_cifar10.py` uses SimpleCIFARNet, Kaggle uses ResNet18 → Not comparable

**Option A (Recommended)**: Standardize on ResNet18
```python
# src/experiments/run_cifar10.py
from src.core.models import ResNet18  # Change from SimpleCIFARNet

def main():
    model = ResNet18(num_classes=10)  # Industry standard
    # ... rest unchanged
```

**Option B**: Keep separate but label clearly
- Update docs to distinguish "toy model" vs "benchmark model"
- Ensure no cross-comparisons between architectures

**Deliverables**:
- Consistent architecture across experiments OR
- Clear documentation of architecture choices

---

## 🟡 MEDIUM PRIORITY - Recommended for Strong Accept

### 3. Refactor Monolithic Script (5 days)
**File**: `run_all_kaggle.py` (7,800 lines)

**Target Structure**:
```
kaggle/
├── runners/
│   ├── mnist_runner.py
│   ├── cifar10_runner.py
│   └── nlp_runner.py
├── plotting/
│   ├── loss_curves.py
│   └── heatmaps.py
├── configs/
│   └── experiment_configs.py
└── run_all.py  # < 500 lines
```

**Benefits**:
- Maintainability
- Easier debugging
- Reduced bug surface

**Note**: Not a blocker for publication if time-constrained

---

### 4. SAM Interface Unification (1 day)
**Problem**: 200+ lines of SAM code duplicated in `kaggle/resnet18_cifar10.py`

**Fix**:
```python
# kaggle/resnet18_cifar10.py
# Remove inline SAMSGD, SAMAdam classes
from src.core.pytorch_optimizers import SAMWrapper

# Use unified interface
optimizer = SAMWrapper(
    model.parameters(),
    base_optimizer=torch.optim.SGD,
    rho=0.05,
    lr=0.01
)
```

**Benefits**:
- Single source of truth
- Easier to fix bugs
- Consistent behavior

---

## 🟢 LOW PRIORITY - Nice to Have

All low-priority items already completed! ✅

---

## Quick Verification Before Submission

```bash
# 1. Verify validation split works
python -c "from src.core.data_utils import get_mnist_loaders; \
train, val, test = get_mnist_loaders(val_split=0.1, seed=42); \
print(f'Train: {len(train.dataset)}, Val: {len(val.dataset)}, Test: {len(test.dataset)}')"
# Expected: Train: 54000, Val: 6000, Test: 10000

# 2. Run all tests
pytest tests/ -v
# Expected: All tests pass (193 tests)

# 3. Verify config fairness
pytest tests/test_config_fairness.py -v
# Expected: 10 passed

# 4. Check for zombie configs
python scripts/validate_configs.py
# Expected: Report generated

# 5. Verify dependencies pinned
pip show torch optuna mlflow | grep Version
# Expected: torch==2.6.0, optuna==4.1.0, mlflow==2.19.0
```

---

## Timeline

| Task | Days | Start | End |
|------|------|-------|-----|
| Re-run experiments | 3-5 | Dec 10 | Dec 14 |
| Model standardization | 2 | Dec 15 | Dec 16 |
| Paper updates | 2 | Dec 17 | Dec 18 |
| **Submission Ready** | - | - | **Dec 19** |

**Optional** (if time permits):
| Task | Days | Start | End |
|------|------|-------|-----|
| Refactor monolith | 5 | Dec 19 | Dec 24 |
| SAM unification | 1 | Dec 25 | Dec 25 |

---

## Success Criteria

### Minimum for Publication (WEAK ACCEPT)
- [x] Data leakage fixed ✅
- [x] Dependencies pinned ✅
- [x] Baseline fairness validated ✅
- [ ] Experiments re-run with clean code
- [ ] Model architectures standardized OR documented

### Target for Strong Accept
- [ ] All minimum criteria
- [ ] Code refactored (no 7k line files)
- [ ] SAM interface unified
- [ ] 100% config key usage

---

## Current Status

**Completion**: 67% (8/12 items)  
**Confidence**: High (critical issues resolved)  
**Next Action**: Re-run experiments  
**Blocker**: None (all tools ready)

---

For detailed information, see:
- `docs/FINAL_REMEDIATION_COMPLETE.md` (full summary)
- `docs/TECHNICAL_DEBT_ROADMAP.md` (detailed plan)
- `docs/AUDIT_RESPONSE_CRITICAL_FIXES.md` (audit response)
