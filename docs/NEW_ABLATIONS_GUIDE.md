# New Ablation Studies Integration Guide

## Overview

Three new ablation studies have been added to complete the research suite:
1. **Weight Decay Ablation** - Analyzes regularization effects
2. **Scheduler Ablation** - Compares learning rate scheduling strategies  
3. **Optimizer Comparison Matrix** - All-vs-all statistical comparisons

---

## 1. Weight Decay Ablation

**Purpose**: Systematically test different L2 regularization strengths to find optimal generalization.

**Location**: `src/experiments/weight_decay_ablation.py`

**Tested Values**: 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2

**Key Features**:
- Multi-seed statistical validation
- Generalization gap analysis (train accuracy - test accuracy)
- Dual visualization: accuracy trends + generalization gap
- Statistical comparisons between weight decay values
- Optimal weight decay recommendation

**Usage in run_all_kaggle.py**:
```bash
python run_all_kaggle.py --experiments wd_ablation --seeds 42,123,456
```

**Configuration**:
```python
base_config = {
    'dataset': 'MNIST',
    'model': 'SimpleMLP',
    'lr': 1e-3,
    'epochs': 10,
    'batch_size': 128
}

run_weight_decay_ablation(
    base_config,
    weight_decays=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    optimizers=['SGD', 'SGD_Momentum', 'Adam', 'AdamW'],
    seeds=[42, 123, 456, ...],
    results_dir="results/wd_ablation"
)
```

**Outputs**:
- CSV files: `wd_ablation_summary.csv`, `wd_ablation_generalization.csv`
- Plots: `wd_ablation_accuracy.png`, `wd_ablation_generalization.png`
- Report: `wd_ablation_report.txt`

**Key Insights**:
- Compares test accuracy across different weight decay values
- Identifies weight decay that best balances fit and generalization
- Analyzes generalization gap to detect overfitting
- Provides statistical validation of optimal choice

---

## 2. Scheduler Ablation

**Purpose**: Compare learning rate scheduling strategies and their impact on convergence.

**Location**: `src/experiments/scheduler_ablation.py`

**Tested Schedulers**:
- `None` - Constant learning rate
- `StepLR` - Step decay (gamma=0.5, step_size=5)
- `ExponentialLR` - Exponential decay (gamma=0.9)
- `CosineAnnealingLR` - Cosine annealing (T_max=epochs)

**Key Features**:
- Multi-seed statistical validation
- Convergence epoch tracking (when test accuracy plateaus)
- Dual visualization: final accuracy + convergence speed
- Statistical comparisons between schedulers
- Optimal scheduler recommendation

**Usage in run_all_kaggle.py**:
```bash
python run_all_kaggle.py --experiments scheduler_ablation --seeds 42,123,456
```

**Configuration**:
```python
base_config = {
    'dataset': 'MNIST',
    'model': 'SimpleMLP',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 15,
    'batch_size': 128
}

run_scheduler_ablation(
    base_config,
    schedulers=['None', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR'],
    optimizers=['SGD', 'Adam', 'AdamW'],
    seeds=[42, 123, 456, ...],
    results_dir="results/scheduler_ablation"
)
```

**Outputs**:
- CSV files: `scheduler_ablation_summary.csv`, `scheduler_ablation_convergence.csv`
- Plots: `scheduler_ablation_accuracy.png`, `scheduler_ablation_convergence.png`
- Report: `scheduler_ablation_report.txt`

**Key Insights**:
- Compares final test accuracy across schedulers
- Identifies which scheduler achieves best performance
- Analyzes convergence speed (epochs to plateau)
- Provides statistical validation of scheduler choice

**Convergence Detection**:
Plateau detected when test accuracy improvement < 0.1% over last 3 epochs.

---

## 3. Optimizer Comparison Matrix

**Purpose**: Comprehensive all-vs-all statistical comparison of optimizers.

**Location**: `src/analysis/optimizer_comparison_matrix.py`

**Key Features**:
- Pairwise t-tests between all optimizer pairs
- Cohen's d effect size calculations
- Three heatmap visualizations:
  - **P-value Matrix**: Statistical significance
  - **Effect Size Matrix**: Magnitude of differences
  - **Win/Loss Matrix**: Which optimizer performs better
- Comprehensive text report with rankings
- Multiple comparison correction (Bonferroni)

**Usage in run_all_kaggle.py**:
```bash
# Requires MNIST results to exist first
python run_all_kaggle.py --experiments mnist,optimizer_comparison --seeds 42,123,456
```

**Configuration**:
```python
run_optimizer_comparison_matrix(
    results_dir="results/mnist",
    optimizers=['SGD', 'SGD_Momentum', 'Adam', 'AdamW', 'AMSGrad'],
    metric='test_accuracy',
    output_dir="results/optimizer_comparison",
    alpha=0.05
)
```

**Outputs**:
- CSV files: `pairwise_pvalues.csv`, `pairwise_effect_sizes.csv`, `win_loss_matrix.csv`
- Plots: `pvalue_matrix.png`, `effect_size_matrix.png`, `win_loss_matrix.png`
- Report: `comparison_report.txt`

**Key Insights**:
- Identifies which optimizer pairs have statistically significant differences
- Quantifies effect sizes to determine practical significance
- Provides clear ranking of optimizers
- Shows win/loss record for each optimizer pair
- Applies Bonferroni correction for multiple comparisons

**Interpretation**:
- **P-value < 0.05**: Statistically significant difference
- **Effect Size**:
  - Small: |d| < 0.5
  - Medium: 0.5 ≤ |d| < 0.8
  - Large: |d| ≥ 0.8
- **Win/Loss**: +1 if row > column, -1 if row < column, 0 if not significant

---

## Running All New Ablations

**Quick Test (2 seeds, 2 epochs)**:
```bash
python run_all_kaggle.py --quick \
  --experiments wd_ablation,scheduler_ablation \
  --seeds 42,123
```

**Full Suite (10 seeds, production settings)**:
```bash
python run_all_kaggle.py \
  --experiments wd_ablation,scheduler_ablation,optimizer_comparison \
  --seeds 42,123,456,789,1011,1213,1415,1617,1819,2021
```

**Complete Research Pipeline**:
```bash
# Run everything including new ablations
python run_all_kaggle.py \
  --experiments all \
  --seeds 42,123,456,789,1011,1213,1415,1617,1819,2021
```

---

## Verification

Test the new ablations independently:
```bash
python test_new_ablations.py
```

This runs minimal tests for:
- Weight decay ablation (2 seeds, 2 epochs, 2 WD values)
- Scheduler ablation (2 seeds, 2 epochs, 2 schedulers)
- Optimizer comparison (synthetic data test)

---

## Integration Details

All three studies are now integrated into `run_all_kaggle.py`:

**Added to experiment selection list** (line ~4228):
```python
selected_experiments = ['mnist', 'cifar10', 'nlp', 'medical', '2d', 
                        'robustness', 'sam', 'ablation', 'batch_ablation', 'lr_ablation', 
                        'wd_ablation', 'scheduler_ablation', 'optimizer_comparison', 'resnet', 'highdim']
```

**Execution blocks added**:
- Weight decay ablation: Lines ~4460-4485
- Scheduler ablation: Lines ~4487-4512  
- Optimizer comparison: Lines ~4514-4539

**Quick mode settings**:
- WD ablation: Tests 3 values instead of 6
- Scheduler ablation: Tests 2 schedulers instead of 4
- Both: 2 optimizers instead of 4

---

## Research Paper Integration

These ablations complete the experimental suite needed for publication:

**Hyperparameter Sensitivity Section**:
1. Learning Rate Ablation → Optimal LR identification
2. Batch Size Ablation → Batch size effects on convergence
3. **Weight Decay Ablation** → Regularization impact (NEW)
4. **Scheduler Ablation** → Learning rate scheduling strategies (NEW)

**Statistical Validation Section**:
1. Multi-seed statistical comparisons (existing)
2. Power analysis and effect sizes (existing)
3. **Optimizer Comparison Matrix** → Comprehensive pairwise validation (NEW)

**Tables/Figures to Include**:
- Table: Weight decay vs test accuracy with generalization gap
- Table: Scheduler comparison with convergence epochs
- Figure: WD ablation dual plot (accuracy + gen gap)
- Figure: Scheduler ablation dual plot (accuracy + convergence)
- Figure: Optimizer comparison heatmaps (3 matrices)

---

## Next Steps

1. **Run full experiments** with 10 seeds on all datasets
2. **Analyze results** - check CSV files and statistical reports
3. **Generate figures** - use provided plots for paper
4. **Extract insights** - identify optimal hyperparameters
5. **Write paper sections** - integrate findings into manuscript

---

## Troubleshooting

**Import errors**:
- Ensure `scipy` is installed for statistical tests
- Check Python path includes project root

**Missing results**:
- Optimizer comparison requires MNIST/CIFAR10 results first
- Run base experiments before comparison matrix

**Memory issues**:
- Use `--quick` mode for testing
- Reduce number of seeds or optimizers

**Convergence detection**:
- Adjust thresholds in scheduler_ablation.py if needed
- Default: plateau when improvement < 0.1% over 3 epochs

---

## Files Modified

**New files created**:
1. `src/experiments/weight_decay_ablation.py` (350 lines)
2. `src/experiments/scheduler_ablation.py` (350 lines)
3. `src/analysis/optimizer_comparison_matrix.py` (380 lines)
4. `test_new_ablations.py` (140 lines)
5. `docs/NEW_ABLATIONS_GUIDE.md` (this file)

**Modified files**:
1. `run_all_kaggle.py` - Added 3 new experiment blocks and CLI options

**No files deleted** - All existing functionality preserved!

---

## Quick Reference

| Ablation | Purpose | Outputs | Key Metric |
|----------|---------|---------|------------|
| Weight Decay | Regularization | Accuracy + Gen Gap | Generalization Gap |
| Scheduler | LR Strategy | Accuracy + Convergence | Convergence Epoch |
| Comparison | Statistical Validation | P-values + Effect Sizes | Win/Loss Record |

**Estimated Runtime** (MNIST, 10 seeds):
- Weight Decay: ~15-20 min (6 values × 4 optimizers)
- Scheduler: ~15-20 min (4 schedulers × 3 optimizers)
- Comparison: ~2-3 min (analysis only, uses existing results)

**Total**: ~40 minutes for complete ablation suite on MNIST
