# Technical Debt Roadmap - Post-Audit

**Last Updated**: December 9, 2025  
**Status**: Critical fixes completed, architectural improvements pending

---

## Priority 1: HIGH - Required for Publication (5-7 days)

### 1.1 Re-run All Experiments with Fixed Code ⏱️ 3-5 days
**Reason**: Previous results may contain test set leakage  
**Impact**: All figures, tables, and reported metrics must be regenerated  
**Difficulty**: Low (automation exists)  
**Owner**: Research Team

**Steps**:
```bash
# 1. Clear old contaminated results
mv results results_pre_audit_backup
mkdir results

# 2. Run multi-seed experiments with validation split
python src/experiments/run_multi_seed.py \
  --seeds 42,123,456,789,101112 \
  --config configs/nn_tuning.json \
  --use-validation  # NEW FLAG NEEDED

# 3. Run Kaggle benchmarks (GPU)
cd kaggle
python run_all_kaggle.py --experiments mnist,cifar10,nlp --seeds 42,123,456

# 4. Regenerate all visualizations
python src/visualization/plot_results.py --results-dir ../results
```

**Acceptance Criteria**:
- [ ] All experiments use validation split for tuning
- [ ] Test set accessed only once (final evaluation)
- [ ] Results directory timestamp > Dec 9, 2025
- [ ] Statistical analysis recalculated with new results

---

### 1.2 Baseline Fairness Audit ⏱️ 4-6 hours
**Reason**: Ensure all optimizers get equal hyperparameter search ranges  
**Impact**: Biased comparisons invalidate conclusions  
**Difficulty**: Low (manual review + unit test)  
**Owner**: ML Engineer

**Files to Audit**:
- `configs/nn_tuning.json`
- `configs/cifar10_tuning.json`
- `configs/benchmark_hyperparameters.json`

**Checks**:
```python
# Example: Are LR ranges symmetric?
config = {
    "Adam": {"lr": [1e-4, 1e-3, 1e-2]},      # 3 values
    "SGD":  {"lr": [1e-4, 1e-3, 1e-2]},      # 3 values ✅
    "RMSProp": {"lr": [1e-3]},               # 1 value ❌ UNFAIR
}
```

**Script to Create**:
```python
# tests/test_config_fairness.py
def test_search_space_symmetry():
    """Verify all optimizers get equal search ranges."""
    config = load_config("configs/nn_tuning.json")
    
    lr_ranges = {opt: len(params['lr']) for opt, params in config.items()}
    assert all(count >= 3 for count in lr_ranges.values()), \
        f"All optimizers need ≥3 LR values: {lr_ranges}"
    
    # Check momentum/beta ranges too
    # ...
```

**Acceptance Criteria**:
- [ ] All optimizers have ≥3 learning rates
- [ ] Momentum/beta parameters have ≥3 values where applicable
- [ ] Unit test `test_config_fairness.py` passes
- [ ] Document any intentional asymmetries in `docs/CONFIG_DESIGN.md`

---

### 1.3 Model Architecture Standardization ⏱️ 2 days
**Reason**: `run_cifar10.py` uses SimpleCIFARNet, Kaggle uses ResNet18 → Not comparable  
**Impact**: Results claim "CIFAR-10 benchmark" but test different architectures  
**Difficulty**: Medium (requires re-running some experiments)  
**Owner**: Research Team

**Options**:
1. **Option A (Recommended)**: Standardize on ResNet18 everywhere
   - ✅ Industry standard, published architecture
   - ✅ Already used in Kaggle benchmarks
   - ❌ Slower training (deeper network)

2. **Option B**: Keep separate but clearly label
   - ✅ Fast iteration with SimpleCIFARNet
   - ✅ Keep existing results
   - ❌ Confusing documentation

**Implementation (Option A)**:
```python
# src/experiments/run_cifar10.py
from src.core.models import ResNet18  # Change from SimpleCIFARNet

def main():
    model = ResNet18(num_classes=10)  # Standardize
    # ... rest unchanged
```

**Acceptance Criteria**:
- [ ] All CIFAR-10 experiments use same architecture
- [ ] `README.md` updated to reflect architecture choice
- [ ] Ablation studies verify optimizers work on both shallow and deep nets
- [ ] Documentation clarifies SimpleCIFARNet = "toy model", ResNet18 = "benchmark model"

---

## Priority 2: MEDIUM - Code Quality (5-7 days)

### 2.1 Decompose Monolithic Script (run_all_kaggle.py) ⏱️ 5 days
**Reason**: 7,800 lines in one file → unmaintainable, high bug risk  
**Impact**: Maintainability, not scientific validity  
**Difficulty**: High (requires careful refactoring)  
**Owner**: Software Engineer

**Target Architecture**:
```
kaggle/
├── runners/
│   ├── mnist_runner.py
│   ├── cifar10_runner.py
│   ├── nlp_runner.py
│   └── medical_runner.py
├── plotting/
│   ├── loss_curves.py
│   ├── heatmaps.py
│   └── landscape_3d.py
├── configs/
│   ├── mnist_config.py
│   ├── cifar10_config.py
│   └── nlp_config.py
└── run_all.py  # Orchestrator (< 500 lines)
```

**Migration Strategy**:
1. Extract plotting functions → `plotting/` (day 1)
2. Extract configuration → `configs/` (day 1)
3. Extract experiment runners → `runners/` (day 2-3)
4. Create orchestrator → `run_all.py` (day 4)
5. Test equivalence → compare old vs new results (day 5)

**Acceptance Criteria**:
- [ ] No file > 1,000 lines
- [ ] Original `run_all_kaggle.py` deprecated (moved to `legacy/`)
- [ ] New `run_all.py` produces byte-identical results (checksum match)
- [ ] Documentation updated

---

### 2.2 SAM Interface Unification ⏱️ 1 day
**Reason**: `SAMWrapper` uses closure, inline versions use standard `step()` → code duplication  
**Impact**: 200+ lines duplicated in `kaggle/resnet18_cifar10.py`  
**Difficulty**: Medium (API design)  
**Owner**: ML Engineer

**Current Duplication**:
```python
# kaggle/resnet18_cifar10.py (200 lines)
class SAMSGD(torch.optim.Optimizer):
    def step(self, closure):
        # ... full SAM implementation ...

# src/core/pytorch_optimizers.py (150 lines)
class SAMWrapper(Optimizer):
    def step(self, closure):
        # ... DIFFERENT SAM implementation ...
```

**Proposed Solution**:
```python
# src/core/pytorch_optimizers.py
class SAMWrapper(Optimizer):
    def __init__(self, params, base_optimizer_class, rho=0.05, **base_kwargs):
        """
        Unified SAM wrapper supporting any base optimizer.
        
        Examples:
            SAMWrapper(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.01)
            SAMWrapper(model.parameters(), torch.optim.Adam, rho=0.05, lr=0.001)
        """
        self.base_optimizer = base_optimizer_class(params, **base_kwargs)
        self.rho = rho
    
    def step(self, closure):
        """Requires closure for adversarial gradient computation."""
        # 1. Compute gradients at current point
        loss = closure()
        
        # 2. Compute adversarial perturbation
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grad_norm = torch.norm(p.grad)
                        perturbation = self.rho * p.grad / (grad_norm + 1e-12)
                        p.add_(perturbation)
        
        # 3. Compute adversarial gradients
        self.zero_grad()
        closure()
        
        # 4. Restore and update with adversarial gradients
        # ... implementation ...
        
        return loss
```

**Acceptance Criteria**:
- [ ] `kaggle/` scripts import from `src/core/pytorch_optimizers`
- [ ] No inline SAM implementations (except legacy backup)
- [ ] Unit tests verify SAM behavior identical before/after refactor
- [ ] Benchmark shows <1% performance difference

---

## Priority 3: LOW - Polish (3-5 days)

### 3.1 Zombie Config Detection ⏱️ 4 hours
**Reason**: Unused JSON keys silently ignored → typos cause silent failures  
**Impact**: Debugging difficulty  
**Difficulty**: Low (static analysis)

**Tool to Create**:
```python
# scripts/validate_configs.py
def find_zombie_keys(config_path, usage_script):
    """Find JSON keys never accessed by code."""
    config = json.load(open(config_path))
    script_content = open(usage_script).read()
    
    for key in config.keys():
        if key not in script_content:
            print(f"⚠️ Zombie key: {key} in {config_path}")
```

**Acceptance Criteria**:
- [ ] `scripts/validate_configs.py` detects unused keys
- [ ] CI/CD runs validation on every commit
- [ ] All zombie keys documented or removed

---

### 3.2 Hardware Agnosticism Audit ⏱️ 2 hours
**Reason**: Hardcoded `.cuda()` calls may fail on CPU/MPS  
**Impact**: Portability  
**Difficulty**: Low (grep + fix)

**Search Pattern**:
```bash
grep -r "\.cuda()" src/ kaggle/ --exclude-dir=__pycache__
grep -r "device = 'cuda'" src/ kaggle/
```

**Fix Pattern**:
```python
# Before
x = x.cuda()

# After
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

**Acceptance Criteria**:
- [ ] All scripts run on CPU (slower but functional)
- [ ] Device selection via `--device cpu/cuda/mps` flag
- [ ] CI tests run on CPU

---

### 3.3 Auto-Wiring Safety Check ⏱️ 3 hours
**Reason**: Audit flagged risk of test set access via "auto-wiring" best params  
**Impact**: Data leakage risk  
**Difficulty**: Low (code review)

**Files to Audit**:
```bash
# Search for automatic parameter loading
grep -r "best_params" src/experiments/
grep -r "load_checkpoint.*test" src/
grep -r "mlflow.*test" src/
```

**Safe Pattern**:
```python
# ✅ SAFE: Manual separation
best_params = optuna_study.best_params  # Tuned on validation set
final_model = train(best_params, train_data)
test_acc = evaluate(final_model, test_data)  # Only called once

# ❌ UNSAFE: Auto-wiring
for params in optuna_study.trials:
    test_acc = evaluate(train(params, train_data), test_data)  # Test set in loop!
```

**Acceptance Criteria**:
- [ ] No test set access inside tuning loops
- [ ] Test set accessed only in `final_evaluation()` functions
- [ ] Code review by second researcher

---

## Timeline Summary

| Priority | Task | Days | Blocker |
|----------|------|------|---------|
| 🔴 HIGH | Re-run experiments | 3-5 | - |
| 🔴 HIGH | Baseline fairness | 0.5 | - |
| 🔴 HIGH | Model standardization | 2 | - |
| 🟡 MEDIUM | Decompose monolith | 5 | - |
| 🟡 MEDIUM | SAM unification | 1 | - |
| 🟢 LOW | Zombie configs | 0.5 | - |
| 🟢 LOW | Hardware agnosticism | 0.25 | - |
| 🟢 LOW | Auto-wiring audit | 0.375 | - |

**Total Estimated Effort**: 12-14 days (2-3 weeks with testing/review)

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Re-run results differ significantly | Medium | High | Expected due to test set leakage fix; document differences |
| Refactor breaks existing experiments | Low | High | Version control, checksum validation |
| Timeline slips | Medium | Medium | Prioritize HIGH items only for initial publication |
| New bugs introduced | Low | Medium | Maintain 100% test coverage, add regression tests |

---

## Success Criteria for "Research Grade"

- [x] No data leakage (test set isolated)
- [x] Reproducible (pinned dependencies)
- [x] Statistically valid (corrections applied)
- [ ] Fair comparisons (equal search spaces)
- [ ] Consistent architectures (standardized models)
- [ ] Maintainable code (decomposed monolith)

**Current Status**: 3/6 ✅ (50%)  
**Target for Publication**: 5/6 ✅ (83%)  
**Target for "Strong Accept"**: 6/6 ✅ (100%)

---

## Notes

- This roadmap assumes 1 FTE (full-time equivalent) working on codebase improvements
- HIGH priority items are **blockers** for publication
- MEDIUM/LOW items improve quality but don't invalidate current results (if using fixed code)
- Consider parallelizing: One person on experiments (1.1), another on fairness audit (1.2)
