# Fair Comparison Validity: Preventing Biased Benchmarks

**Senior Principal Software Engineer — Scientific Integrity Framework**  
**Date:** January 6, 2026  
**Purpose:** Establish rules for valid optimizer comparisons and prevent strawman benchmarking

---

## Executive Summary

This document defines **five critical fairness constraints** for optimizer benchmarking:

1. **Search Budget Parity** — All optimizers receive equal hyperparameter tuning effort
2. **System Overhead Isolation** — Never compare wall-clock times across different task types
3. **Controlled Variables** — Only change optimizer; fix dataset, architecture, batch size, etc.
4. **Iteration vs. Epoch Consistency** — Use correct units for each comparison type
5. **Statistical Rigor** — Multi-seed experiments with proper significance testing

**Purpose:** Prevent "Optimizer A is better" claims that are actually artifacts of unfair experimental design.

---

## 1. Search Budget Parity (Automated Validation)

### The Strawman Comparison

**Unfair Experiment:**
```python
# SGD: Test 10 learning rates
sgd_lrs = [1e-4, 1e-3, 1e-2, 0.1]  # 4 trials

# Adam: Test 120 learning rates + betas
adam_lrs = [1e-5, 3e-5, 1e-4, 3e-4, ...]  # 30 LR values
adam_betas = [(0.9, 0.999), (0.9, 0.99), (0.8, 0.999), ...]  # 4 beta pairs
# Total: 30 * 4 = 120 trials

# Result: Adam achieves 92% test accuracy, SGD achieves 89%
# Conclusion: "Adam is better" ❌ FALSE
# Reality: Adam had 30× more tuning effort
```

### Codebase Protection

**File:** `scripts/check_search_budget_parity.py`

**How It Works:**
```python
def compute_grid_size(sweep_config):
    """Compute total number of hyperparameter combinations."""
    size = 1
    for param, values in sweep_config.items():
        size *= len(values)
    return size

def check_search_budget_parity(config_path, threshold=5.0):
    """
    Verify search budgets are balanced across optimizers.
    
    Returns:
        valid (bool): True if max_budget / min_budget <= threshold
        ratios (dict): Per-optimizer search budget sizes
    """
    config = load_config(config_path)
    budgets = {}
    
    for optimizer, sweep_params in config['hyperparameter_sweep'].items():
        budgets[optimizer] = compute_grid_size(sweep_params)
    
    max_budget = max(budgets.values())
    min_budget = min(budgets.values())
    ratio = max_budget / min_budget
    
    if ratio > threshold:
        print(f"❌ UNFAIR: Search budget ratio {ratio:.1f}x exceeds threshold {threshold}x")
        print(f"   Max budget: {max_budget} ({max(budgets, key=budgets.get)})")
        print(f"   Min budget: {min_budget} ({min(budgets, key=budgets.get)})")
        return {'valid': False, 'ratio': ratio, 'budgets': budgets}
    else:
        print(f"✅ FAIR: Search budget ratio {ratio:.1f}x within threshold")
        return {'valid': True, 'ratio': ratio, 'budgets': budgets}
```

### Usage Example

**Config File:** `configs/nn_tuning.json`
```json
{
  "hyperparameter_sweep": {
    "sgd": {
      "lr": [0.001, 0.003, 0.01, 0.03, 0.1],
      "momentum": [0.0, 0.5, 0.9, 0.99],
      "weight_decay": [0.0, 1e-4, 1e-3]
    },
    "adam": {
      "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
      "beta1": [0.8, 0.9, 0.95],
      "beta2": [0.99, 0.999, 0.9999],
      "weight_decay": [0.0, 1e-4, 1e-3]
    }
  }
}
```

**Validation:**
```bash
python scripts/check_search_budget_parity.py --config configs/nn_tuning.json

# Output:
# SGD search budget: 5 * 4 * 3 = 60 trials
# Adam search budget: 5 * 3 * 3 * 3 = 135 trials
# Ratio: 135 / 60 = 2.25x
# ✅ FAIR: Search budget ratio 2.25x within threshold 5.0x
```

### Thesis Documentation

**Methodology Section (Required):**

> **2.6 Search Budget Parity Validation**
>
> To prevent bias from unequal hyperparameter tuning effort, we enforce search budget parity: the ratio of grid sizes (number of hyperparameter combinations) between optimizers must not exceed 5×.
>
> For example, in our CIFAR-10 ResNet-18 benchmark:
> - SGD: 5 LRs × 4 momentums × 3 weight decays = 60 trials
> - Adam: 5 LRs × 3 beta1s × 3 beta2s × 3 weight decays = 135 trials
> - Ratio: 135/60 = 2.25× (within threshold ✓)
>
> All configurations passed this fairness check (Appendix B.1). This ensures observed performance differences reflect algorithmic properties, not unequal search effort.

---

## 2. System Overhead Isolation (Never Cross-Compare Task Types)

### The Invalid Comparison

**Logically Flawed Experiment:**
```python
# Measure wall-clock time for 1000 optimizer steps
rosenbrock_2d_time = 0.5 seconds
resnet18_cifar10_time = 120 seconds

# Conclusion: "ResNet-18 is 240× slower" ❌ MEANINGLESS
```

**Why This is Wrong:**
- 2D function: Pure NumPy (10 lines of code, 2 parameters)
- ResNet-18: DataLoader, GPU transfer, autograd, backprop through 18 layers (11M parameters)

**What You're Actually Measuring:** System engineering overhead, NOT optimizer performance.

### Valid Comparisons

#### ✅ Valid: Same Task, Different Optimizers
```python
# All on ResNet-18 CIFAR-10
sgd_time_to_90pct_acc = 180 seconds
adam_time_to_90pct_acc = 150 seconds
# Conclusion: "Adam is 20% faster for this task" ✓ VALID
```

#### ✅ Valid: Same Optimizer, Different Tasks (Scaling Analysis)
```python
# SGD on increasing problem sizes
sgd_rosenbrock_2d = 0.5 seconds
sgd_rastrigin_100d = 5 seconds
sgd_resnet18 = 180 seconds
# Conclusion: "SGD scales sublinearly with dimension" ✓ VALID (different analysis type)
```

#### ❌ Invalid: Cross-Task, Cross-Optimizer
```python
# Comparing oranges to spaceships
sgd_rosenbrock = 0.5 seconds
adam_resnet18 = 150 seconds
# Conclusion: "Adam is slower" ❌ NONSENSICAL
```

### Codebase Structure

**Enforced Separation:**
```
results/
├── 2d_functions/
│   ├── sgd_rosenbrock.csv
│   ├── adam_rosenbrock.csv
│   └── comparison_plot.png  # Valid: Same task, different optimizers
│
├── neural_networks/
│   ├── sgd_resnet18_cifar10.csv
│   ├── adam_resnet18_cifar10.csv
│   └── comparison_plot.png  # Valid: Same task, different optimizers
│
└── cross_task_scaling/
    ├── sgd_across_tasks.csv
    └── scaling_analysis.png  # Valid: Same optimizer, different tasks (for scaling study)
```

### Thesis Documentation

**Results Section (Mandatory Disclaimer):**

> **4.1 Optimizer Comparison Rules**
>
> We compare optimizers **only within the same task category**:
> - **2D Functions:** Figure 4.1 compares SGD/Adam/Momentum on Rosenbrock
> - **Neural Networks:** Figure 4.5 compares SGD/Adam/AdamW on ResNet-18 CIFAR-10
>
> **Invalid Comparison (Avoided):** We do NOT compare "Rosenbrock wall-clock time" vs "ResNet-18 wall-clock time" across optimizers, as this would measure system engineering overhead rather than algorithmic performance.
>
> **Scaling Analysis (Separate):** Figure 4.9 shows how a single optimizer (SGD) scales across problem dimensions (2D, 100D, 11M-D), which is a valid within-optimizer scaling study.

---

## 3. Controlled Variables (Scientific Method 101)

### The Confounded Experiment

**Bad Design:**
```python
# "Compare" SGD vs Adam
sgd_experiment = {
    'optimizer': 'sgd',
    'dataset': 'cifar10',
    'batch_size': 128,
    'model': 'resnet18',
    'data_augmentation': True,
    'lr_scheduler': 'step',
}

adam_experiment = {
    'optimizer': 'adam',
    'dataset': 'cifar100',  # ❌ Changed dataset
    'batch_size': 256,  # ❌ Changed batch size
    'model': 'resnet50',  # ❌ Changed model
    'data_augmentation': False,  # ❌ Changed augmentation
    'lr_scheduler': 'cosine',  # ❌ Changed scheduler
}

# Result: Adam gets 75% accuracy, SGD gets 68%
# Conclusion: "Adam is better" ❌ FALSE
# Reality: You changed 5 variables simultaneously
```

### Scientific Control

**Correct Design:**
```python
# Fixed experimental conditions
base_config = {
    'dataset': 'cifar10',
    'batch_size': 128,
    'model': 'resnet18',
    'data_augmentation': True,
    'lr_scheduler': 'cosine',
    'epochs': 90,
    'seeds': [42, 123, 456],
}

# Only change: optimizer
sgd_experiment = {**base_config, 'optimizer': 'sgd', 'lr': 0.01}
adam_experiment = {**base_config, 'optimizer': 'adam', 'lr': 0.001}

# NOW you can attribute differences to the optimizer
```

### Codebase Implementation

**File:** `src/experiments/run_nn_experiment.py`

**Expected Pattern:**
```python
def run_optimizer_comparison(base_config, optimizers):
    """
    Run controlled experiment varying only the optimizer.
    
    Args:
        base_config: Fixed experimental settings
        optimizers: List of optimizer configs with tuned hyperparameters
    
    Returns:
        results: DataFrame with optimizer as the only varying factor
    """
    results = []
    for opt_config in optimizers:
        # Merge: base settings + optimizer-specific hyperparams
        full_config = {**base_config, **opt_config}
        
        # Run experiment
        metrics = train_and_evaluate(full_config)
        results.append({
            'optimizer': opt_config['name'],
            'test_accuracy': metrics['test_acc'],
            'train_loss': metrics['final_train_loss'],
            'wall_time': metrics['total_time'],
        })
    
    return pd.DataFrame(results)
```

### Thesis Documentation

**Methodology Section:**

> **2.8 Controlled Variable Protocol**
>
> When comparing optimizers, we fix all experimental conditions:
> - Same dataset (CIFAR-10, 50K train / 10K test)
> - Same architecture (ResNet-18, 11M parameters)
> - Same batch size (128)
> - Same data augmentation (RandomCrop + HorizontalFlip)
> - Same learning rate scheduler (CosineAnnealingLR, T_max=90)
> - Same random seeds ([42, 123, 456] for statistical validity)
>
> **Only variable changed:** Optimizer type and its hyperparameters (tuned independently per optimizer via Optuna, Section 2.5).
>
> This ensures observed performance differences are attributable to the optimizer algorithm, not confounding factors.

---

## 4. Iteration vs. Epoch Consistency (Unit Correctness)

### The Unit Mismatch Error

**Incorrect Graph:**
```python
# X-axis: Epochs (1 epoch = 391 steps for CIFAR-10)
plt.plot(epochs, train_loss, label='Measured')
plt.plot(epochs, C / epochs, label='O(1/k) theory', linestyle='--')  # ❌ WRONG UNITS

# Problem: Theory predicts O(1/k) where k = iterations (steps), not epochs
# For CIFAR-10: Epoch 10 = 3910 steps
# The curve should be: C / (epochs * 391)
```

**Correct Graph:**
```python
# Convert epochs to iterations for theory comparison
iterations = epochs * steps_per_epoch  # steps_per_epoch = 50000 / batch_size
plt.plot(iterations, train_loss, label='Measured')
plt.plot(iterations, C / iterations, label='O(1/k) theory', linestyle='--')  # ✓ CORRECT
plt.xlabel('Iterations (k)')
```

### Codebase Verification

**File:** `src/experiments/theory_practice_validation.py`

**Expected Implementation:**
```python
def fit_convergence_rate(loss_history, rate_type='sublinear'):
    """
    Fit theoretical convergence curve to measured loss.
    
    Args:
        loss_history: Array of losses indexed by ITERATION (not epoch)
        rate_type: 'sublinear' (O(1/k)), 'linear' (O(ρ^k)), etc.
    
    Returns:
        fitted_curve: Theory prediction at each iteration
        rate_constant: Estimated constant C in O(1/k)
    """
    k = np.arange(1, len(loss_history) + 1)  # Iteration count (1-indexed)
    
    if rate_type == 'sublinear':
        # Fit: loss = L_min + C/k
        def model(k, L_min, C):
            return L_min + C / k
        params, _ = curve_fit(model, k, loss_history)
        fitted_curve = model(k, *params)
    
    return fitted_curve, params
```

### Thesis Documentation

**Results Section (Figure Caption):**

> **Figure 4.2:** Training loss convergence for SGD on CIFAR-10 ResNet-18.  
> **X-axis:** Iterations (k), not epochs. For CIFAR-10 with batch size 128, 1 epoch = 391 iterations.  
> **Theory Curve:** Best-fit O(1/k) with C=15.2, L_∞=0.03. The measured convergence closely follows the theoretical prediction up to iteration 10,000 (epoch 25), after which the learning rate decay (StepLR) causes faster-than-O(1/k) convergence.

---

## 5. Statistical Rigor (Multi-Seed Experiments)

### The Reproducibility Crisis

**Unreliable Result:**
```python
# Single random seed
seed = 42
sgd_acc = train_model('sgd', seed=seed)  # 91.2%
adam_acc = train_model('adam', seed=seed)  # 91.5%

# Conclusion: "Adam is 0.3% better" ❌ NOT STATISTICALLY SIGNIFICANT
# Reality: Could be random noise (different initializations, mini-batch sampling)
```

### Correct Approach

**Multi-Seed Protocol:**
```python
seeds = [42, 123, 456, 789, 1011]  # 5 independent runs

sgd_accs = [train_model('sgd', seed=s) for s in seeds]
adam_accs = [train_model('adam', seed=s) for s in seeds]

# Compute statistics
sgd_mean = np.mean(sgd_accs)  # 91.3%
sgd_std = np.std(sgd_accs)  # 0.4%
adam_mean = np.mean(adam_accs)  # 91.8%
adam_std = np.std(adam_accs)  # 0.3%

# Statistical significance test (t-test)
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(sgd_accs, adam_accs)

if p_value < 0.05:
    print(f"Adam is significantly better (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

### Codebase Implementation

**File:** `src/analysis/statistical_tests.py`

**Expected Functions:**
```python
def compare_optimizers_with_significance(results_df, metric='test_accuracy'):
    """
    Compare optimizer performance with statistical tests.
    
    Returns:
        comparison_table: DataFrame with mean, std, p-values, effect sizes
    """
    optimizers = results_df['optimizer'].unique()
    comparison = []
    
    for opt_a, opt_b in itertools.combinations(optimizers, 2):
        values_a = results_df[results_df['optimizer'] == opt_a][metric]
        values_b = results_df[results_df['optimizer'] == opt_b][metric]
        
        # T-test
        t_stat, p_value = ttest_ind(values_a, values_b)
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(values_a) - np.mean(values_b)) / np.sqrt((np.std(values_a)**2 + np.std(values_b)**2) / 2)
        
        comparison.append({
            'optimizer_a': opt_a,
            'optimizer_b': opt_b,
            'mean_a': np.mean(values_a),
            'mean_b': np.mean(values_b),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': effect_size,
        })
    
    return pd.DataFrame(comparison)
```

### Thesis Documentation

**Methodology Section:**

> **2.9 Statistical Validation**
>
> We run each experiment with **5 independent random seeds** ([42, 123, 456, 789, 1011]) to account for initialization and mini-batch sampling variability.
>
> **Statistical Tests:**
> - **Paired t-test** for pairwise optimizer comparison (α = 0.05)
> - **Effect size** (Cohen's d) to quantify practical significance
> - **Error bars** in all plots show mean ± 1 standard deviation
>
> **Significance Reporting:** We report a result as "Optimizer A is better than B" only if:
> 1. p-value < 0.05 (statistically significant)
> 2. |effect_size| > 0.5 (medium or large practical effect)
>
> Results failing these criteria are reported as "no significant difference."

**Results Section (Table Format):**

| Optimizer | Test Accuracy (%) | Std Dev | vs. SGD (p-value) | Effect Size |
|-----------|-------------------|---------|-------------------|-------------|
| SGD       | 91.3 ± 0.4        | 0.4     | —                 | —           |
| Momentum  | 91.8 ± 0.3        | 0.3     | 0.03 *            | 1.4 (large) |
| Adam      | 92.1 ± 0.5        | 0.5     | 0.01 *            | 1.6 (large) |
| AdamW     | 92.3 ± 0.4        | 0.4     | 0.005 **          | 2.1 (large) |

**Interpretation:** AdamW significantly outperforms SGD (p=0.005, Cohen's d=2.1). Momentum improvement over SGD is smaller (d=1.4) but still statistically significant (p=0.03).

---

## Summary: Fair Comparison Checklist

Before claiming "Optimizer X is better than Optimizer Y," verify:

### ✅ Search Budget Parity
- [ ] Run `python scripts/check_search_budget_parity.py --config <config.json>`
- [ ] Ratio < 5.0× (documented in thesis)

### ✅ System Overhead Isolation
- [ ] Only compare optimizers within the same task (2D vs 2D, NN vs NN)
- [ ] Never compare wall-clock times across different task types

### ✅ Controlled Variables
- [ ] Fixed: dataset, model, batch size, scheduler, augmentation, seeds
- [ ] Varying: optimizer type + optimizer-specific hyperparameters only

### ✅ Iteration vs. Epoch Consistency
- [ ] Theory comparison plots use iterations (k), not epochs
- [ ] Conversion factor documented: 1 epoch = (dataset_size / batch_size) steps

### ✅ Statistical Rigor
- [ ] Minimum 5 seeds per experiment
- [ ] Report mean ± std in all tables/figures
- [ ] Run t-tests and compute effect sizes
- [ ] Only claim "better" if p < 0.05 AND |d| > 0.5

---

## Defense Preparation

### Q: "Your Adam results are 1.5% better than SGD. Is that significant?"

**A:** "Yes. We ran 5 independent seeds and performed a paired t-test (p=0.01, Table 4.2). The effect size is Cohen's d=1.6, which is considered 'large' in educational statistics. Additionally, all 5 Adam runs outperformed all 5 SGD runs, suggesting the difference is robust."

---

### Q: "Did you tune hyperparameters equally for all optimizers?"

**A:** "Yes. We verified search budget parity using an automated script (Section 2.6). SGD received 60 hyperparameter configurations, Adam received 135 configurations (ratio 2.25×, within our 5× threshold). The results passed our fairness check (Appendix B.1)."

---

### Q: "Why is your ResNet-18 training time 100× slower than your Rosenbrock experiment?"

**A:** "We do not make cross-task time comparisons. Rosenbrock is a 2-parameter function with analytical gradients; ResNet-18 is an 11M-parameter network with data loading, GPU transfers, and autograd overhead. These are architecturally different tasks. We only compare optimizers within the same task (Figure 4.5 compares SGD vs Adam on ResNet-18, which is valid)."

---

## Conclusion

The difference between a weak thesis ("I ran some experiments") and a strong thesis ("I performed rigorous scientific comparison") is **documented experimental controls**.

These five fairness constraints (search parity, overhead isolation, controlled variables, unit consistency, statistical rigor) are your **defense armor**. Implement them, document them, and no reviewer can question the validity of your comparisons.
