#  Quick Reference - What's New & How to Use

This document provides a quick overview of the major improvements and how to use them.

---

##  What's New (Major Improvements)

### 1. Multi-Seed Experiments 
**Before**: Single seed, unreliable results  
**After**: Multiple seeds with statistics (mean ± std)

```bash
# Run with 5 seeds
python run_full_analysis.py --seeds 1,2,3,4,5
```

**Output**: `97.50 ± 0.15% (n=5)` instead of just `97.5%`

---

### 2. Statistical Analysis 
**Before**: No way to claim one optimizer is better  
**After**: T-tests, p-values, effect sizes, confidence intervals

**Example Result**:
```
AdamW vs SGD+Momentum:
  t-statistic: 2.58
  p-value: 0.032 < 0.05  SIGNIFICANT
  Effect size: 1.83 (large)
  → AdamW is statistically significantly better
```

---

### 3. Error Bar Plots 
**Before**: No error bars, looked unprofessional  
**After**: All plots have mean ± std bands

**Files**: `plot_results.py` now has:
- `plot_multiseed_comparison()` - curves with error bands
- `plot_final_metric_comparison()` - bar charts with error bars
## Publication add-ons (MNIST)

- After running the Kaggle MNIST suite (see `kaggle/mnist_publication/README.md`), aggregate and plot NN ablation:
    - Summaries: `results/nn_ablation_summary.csv`
    - Plots: `plots/nn_ablation_accuracy.png` and `plots/nn_ablation_loss.png`
    - Script: `scripts/run_nn_ablation.py`

- Generate statistical report with normality diagnostics, paired tests and Holm–Bonferroni:
    - CSV: `results/nn_statistical_comparisons.csv`
    - Markdown: `results/nn_statistical_report.md`
    - Script: `scripts/generate_statistical_report.py`

## Outline mapping
See `docs/OUTLINE_MAPPING.md` for how each outline section maps to code, data, and figures.

---

### 4. Unit Tests 
**Before**: 0 tests, gradients unverified  
**After**: 35 tests (100% passing)

```bash
# Run all tests
pytest tests/ -v

# Expected output:
# 35 passed in 0.14s 
```

**What's tested**:
- All gradients verified numerically (1e-5 tolerance)
- All optimizers mathematically correct
- Hessians verified

---

### 5. Ablation Study 
**Before**: Just compared full optimizers  
**After**: Test each component in isolation

```bash
python run_ablation_study.py
```

**Tests**:
1. SGD baseline (no momentum, no adaptive LR)
2. SGD + Momentum only
3. RMSProp (adaptive LR only)
4. Adam (full)
5. Adam + L2 regularization
6. AdamW (decoupled weight decay)

**Quantifies** how much each component contributes!

---

### 6. Baseline Comparison 
**Before**: No comparison with PyTorch  
**After**: Compare custom vs PyTorch built-ins

```bash
python run_baseline_comparison.py
```

**Compares**:
- Custom Adam vs torch.optim.Adam
- Custom AdamW vs torch.optim.AdamW
- Custom SGD+Momentum vs torch.optim.SGD
- Custom RMSProp vs torch.optim.RMSProp

**Validates** implementations are correct!

---

### 7. Input Validation 
**Before**: Crashes on bad input  
**After**: Helpful error messages

```python
from validation import validate_config

config = {'lr': -0.01}  # Invalid!
validate_config(config)
# ValidationError: Learning rate must be positive, got -0.01
```

---

### 8. CIFAR-10 Support 
**Before**: Only MNIST  
**After**: MNIST + CIFAR-10 with ConvNet model

```bash
python run_full_analysis.py --config configs/cifar10_tuning.json
```

---

##  Quick Command Reference

### Testing
```bash
# Run all tests
pytest tests/ -v

# Test gradients only
pytest tests/test_gradients.py -v

# Test optimizers only  
pytest tests/test_optimizers.py -v
```

### Multi-Seed Experiments
```bash
# Quick test (3 seeds)
python run_full_analysis.py --seeds 1,2,3

# Production run (5+ seeds)
python run_full_analysis.py \
    --seeds 1,2,3,4,5 \
    --compare AdamW-SGDMomentum,Adam-RMSProp

# CIFAR-10
python run_full_analysis.py \
    --config configs/cifar10_tuning.json \
    --seeds 1,2,3,4,5
```

### Ablation Study
```bash
# Run component-wise analysis
python run_ablation_study.py

# Output:
# - Bar plot showing contribution of each component
# - Statistical comparison vs baseline
# - Quantifies improvement from momentum, adaptive LR, etc.
```

### Baseline Comparison
```bash
# Compare with PyTorch implementations
python run_baseline_comparison.py

# Output:
# - Statistical tests (custom vs PyTorch)
# - Bar chart comparison
# - Validates correctness
```

### Traditional Pipeline (Single Seed)
```bash
# Full pipeline
python run_all.py

# Quick mode
python run_all.py --quick

# Skip phases
python run_all.py --skip-2d --skip-tuning
```

---

##  Understanding Output

### Multi-Seed Results
**Format**: `mean ± std (n=X)`

**Example**:
```
Test Accuracy: 97.50 ± 0.15% (n=5)
```

**Interpretation**:
- Mean accuracy: 97.50%
- Standard deviation: 0.15%
- Number of seeds: 5
- **95% CI**: approximately [97.20%, 97.80%]

### Statistical Tests
**p-value < 0.05**: Statistically significant difference  
**p-value ≥ 0.05**: No significant difference

**Effect size (Cohen's d)**:
- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- |d| ≥ 0.8: Large

**Example**:
```
p = 0.032 < 0.05  SIGNIFICANT
d = 1.83 → Large effect size
→ Strong evidence that AdamW is better
```

### Error Bars
**Shaded bands**: ±1 std around mean  
**Individual dots**: Results from each seed

**If bands don't overlap**: Strong evidence of difference  
**If bands overlap heavily**: May not be significantly different

---

##  File Organization

### Results Directory
```
results/
 multi-seed results: *_seed1.csv, *_seed2.csv, ...
 aggregated_results.json (statistics)
 ablation/ (ablation study results)
 baselines/ (baseline comparison results)
```

### Plots Directory
```
plots/
 multiseed_comparison_*.png (curves with error bars)
 final_*_comparison.png (bar charts)
 statistical_*.png (comparison plots)
 ablation_study.png (ablation results)
 baseline_comparison.png (custom vs PyTorch)
```

---

##  Best Practices

### Running Experiments
1. **Always use ≥5 seeds** for final results
2. **Use 3 seeds** for quick testing
3. **Report full statistics**: mean, std, n, CI
4. **Include error bars** on all plots
5. **Perform statistical tests** before claiming superiority

### Interpreting Results
1. **Check p-values**: p < 0.05 for significance
2. **Check effect sizes**: Large effect = practical significance
3. **Check confidence intervals**: Non-overlapping = strong difference
4. **Consider variance**: Large std = less reliable

### Documentation
1. **Always report n** (number of seeds)
2. **State assumptions** clearly
3. **Document limitations** (see LIMITATIONS.md)
4. **Include statistical tests** in conclusions

---

##  Common Workflows

### Workflow 1: Quick Test
```bash
# 1. Verify installation
pytest tests/ -v

# 2. Quick multi-seed test (3 seeds)
python run_full_analysis.py --seeds 1,2,3

# Time: ~5-10 minutes
```

### Workflow 2: Full Evaluation
```bash
# 1. Multi-seed experiments (5 seeds)
python run_full_analysis.py \
    --seeds 1,2,3,4,5 \
    --compare AdamW-SGDMomentum

# 2. Ablation study
python run_ablation_study.py

# 3. Baseline comparison
python run_baseline_comparison.py

# Time: ~2-3 hours
```

### Workflow 3: Publication-Ready
```bash
# 1. Multi-seed MNIST (10 seeds)
python run_full_analysis.py \
    --config configs/nn_tuning.json \
    --seeds 1,2,3,4,5,6,7,8,9,10 \
    --compare AdamW-SGDMomentum,Adam-RMSProp,AdamW-Adam

# 2. Multi-seed CIFAR-10 (10 seeds)
python run_full_analysis.py \
    --config configs/cifar10_tuning.json \
    --seeds 1,2,3,4,5,6,7,8,9,10 \
    --compare AdamW-SGDMomentum

# 3. Ablation study
python run_ablation_study.py

# 4. Baseline comparison
python run_baseline_comparison.py

# Time: ~8-12 hours
```

---

##  Documentation Guide

### For Beginners:
1. **README.md** - Start here!
2. **MULTISEED_GUIDE.md** - How to run multi-seed experiments
3. **Quick examples** in each script's `if __name__ == '__main__'`

### For Researchers:
1. **CRITICAL_VALIDATION_REPORT.md** - Scientific validation
2. **LIMITATIONS.md** - Known limitations & assumptions
3. **IMPROVEMENT_PROGRESS.md** - What was fixed and why

### For Developers:
1. **tests/** - Unit test examples
2. **validation.py** - Input validation patterns
3. **Code comments** - Inline documentation

---

##  Troubleshooting

### Problem: Tests failing
```bash
# Re-run with verbose output
pytest tests/ -v --tb=short

# Check specific test
pytest tests/test_gradients.py::TestRosenbrockGradients -v
```

### Problem: "No module named scipy"
```bash
pip install scipy
```

### Problem: "No module named pytest"
```bash
pip install pytest
```

### Problem: Out of memory
```bash
# Reduce batch size in config
# Use fewer seeds
# Run on smaller model (SimpleMLP instead of ConvNet)
```

### Problem: Plots not showing error bars
```bash
# Make sure you're using multi-seed results
# Check that you have multiple CSV files with different seeds
# Use plot_multiseed_comparison() function
```

---

##  Checklist for Publication

Before publishing results:

- [ ] Run with ≥5 seeds
- [ ] All plots have error bars
- [ ] Statistical tests performed
- [ ] p-values and effect sizes reported
- [ ] Confidence intervals computed
- [ ] Unit tests passing (pytest)
- [ ] Ablation study completed
- [ ] Baseline comparison done
- [ ] Limitations documented
- [ ] Results reproducible (seeds recorded)

---

##  Getting Help

1. **Check documentation**: MULTISEED_GUIDE.md, LIMITATIONS.md
2. **Run tests**: `pytest tests/ -v` to verify installation
3. **Check examples**: Look at `if __name__ == '__main__'` blocks
4. **Validate config**: Use `validation.validate_config(config)`

---

**Last Updated**: Current Session  
**Status**: All improvements complete   
**Version**: 2.0 (Major upgrade from 1.0)
