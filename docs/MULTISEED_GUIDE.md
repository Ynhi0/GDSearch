# Multi-Seed Experiments & Statistical Analysis

This document explains how to use the new multi-seed experiment framework and statistical analysis tools.

## Overview

The framework now supports:
-  **Multi-seed experiments** for statistical reliability
-  **Automated statistical tests** (t-tests, effect sizes, confidence intervals)
-  **Error bar visualization** (mean ± std bands)
-  **Comprehensive unit tests** (gradients and optimizers)

## Quick Start

### 1. Run Multi-Seed Experiments

Run experiments with 5 different random seeds:

```bash
python run_multi_seed.py --seeds 1,2,3,4,5
```

This will:
- Run your config with seeds [1, 2, 3, 4, 5]
- Save results as `*_seed1.csv`, `*_seed2.csv`, etc.
- Compute statistics: mean ± std (n=5)

### 2. Full Analysis Pipeline

Run experiments + statistical analysis + plots with one command:

```bash
python run_full_analysis.py \
    --config configs/mnist_tuning.json \
    --seeds 1,2,3,4,5 \
    --compare AdamW-SGDMomentum,Adam-RMSProp
```

This will:
1. **Run multi-seed experiments** (Phase 1)
2. **Aggregate results** with statistics (Phase 2)
3. **Perform t-tests** between optimizer pairs (Phase 3)
4. **Generate plots with error bars** (Phase 4)

### 3. Statistical Comparison

Compare two optimizers with statistical tests:

```python
import numpy as np
from statistical_analysis import compare_optimizers_ttest, print_ttest_results

# Load your results (final test_accuracy from 5 seeds)
adamw_results = np.array([0.975, 0.976, 0.974, 0.977, 0.973])
sgdm_results = np.array([0.971, 0.970, 0.972, 0.969, 0.971])

# Perform t-test
result = compare_optimizers_ttest(
    adamw_results, 
    sgdm_results,
    name_A="AdamW",
    name_B="SGD+Momentum",
    metric="test_accuracy"
)

print_ttest_results(result)
```

Output:
```
======================================================================
Statistical Comparison: AdamW vs SGD+Momentum
Metric: test_accuracy
======================================================================

AdamW:
  Mean: 0.9750
  Std:  0.0015
  N:    5
  95% CI: [0.9726, 0.9774]

SGD+Momentum:
  Mean: 0.9706
  Std:  0.0011
  N:    5
  95% CI: [0.9688, 0.9724]


Test Statistics:
  t-statistic: 4.8137
  p-value:     0.0008
  Significant:  YES (α=0.05)
  Effect size (Cohen's d): 3.4052
  Effect size interpretation: large


 CONCLUSION: AdamW is statistically significantly better
   (p=0.0008 < 0.05, effect size=large)
======================================================================
```

### 4. Visualization with Error Bars

Plot comparison with confidence bands:

```python
from plot_results import plot_multiseed_comparison, plot_final_metric_comparison
import pandas as pd

# Load multi-seed results
results = {
    'AdamW': [
        pd.read_csv('results/AdamW_seed1.csv'),
        pd.read_csv('results/AdamW_seed2.csv'),
        pd.read_csv('results/AdamW_seed3.csv'),
    ],
    'SGD+Momentum': [
        pd.read_csv('results/SGDMomentum_seed1.csv'),
        pd.read_csv('results/SGDMomentum_seed2.csv'),
        pd.read_csv('results/SGDMomentum_seed3.csv'),
    ]
}

# Plot curves with error bands
plot_multiseed_comparison(
    results,
    metric='test_accuracy',
    title='Test Accuracy Comparison (Multi-Seed)',
    save_path='plots/comparison_accuracy.png'
)

# Plot final values as bar chart
plot_final_metric_comparison(
    results,
    metric='test_accuracy',
    title='Final Test Accuracy',
    save_path='plots/final_accuracy.png'
)
```

## Unit Tests

Run all tests (gradients + optimizers):

```bash
pytest tests/ -v
```

Run specific test file:

```bash
# Test gradients only
pytest tests/test_gradients.py -v

# Test optimizers only
pytest tests/test_optimizers.py -v
```

## Project Structure

```
GDSearch/
 run_multi_seed.py           # Multi-seed experiment framework
 run_full_analysis.py        # Full pipeline (experiments + stats + plots)
 statistical_analysis.py     # T-tests, effect sizes, CI computation
 plot_results.py             # Plotting (now with error bars!)

 tests/
    test_gradients.py       # Numerical gradient verification
    test_optimizers.py      # Optimizer correctness tests

 configs/
    mnist_tuning.json       # MNIST experiments
    rosenbrock_tuning.json  # Test function experiments

 results/                    # Multi-seed results stored here
     AdamW_seed1.csv
     AdamW_seed2.csv
     ...
```

## Statistical Validity Checklist

 **Multi-seed experiments**: Always use ≥5 seeds  
 **Report statistics**: Always report "mean ± std (n=X)"  
 **Statistical tests**: Use t-tests to compare optimizers  
 **Effect sizes**: Report Cohen's d for practical significance  
 **Confidence intervals**: Show 95% CI for all means  
 **Error bars**: All plots must show ±1 std bands  
 **Unit tests**: All gradients and optimizers verified numerically  

## Example Workflows

### Workflow 1: Quick Test on MNIST

```bash
# Run 3 seeds quickly (for testing)
python run_full_analysis.py \
    --config configs/mnist_tuning.json \
    --seeds 1,2,3 \
    --compare AdamW-SGDMomentum
```

### Workflow 2: Full Evaluation (5+ seeds)

```bash
# Run 5 seeds for publication-ready results
python run_full_analysis.py \
    --config configs/mnist_tuning.json \
    --seeds 1,2,3,4,5 \
    --compare AdamW-SGDMomentum,Adam-RMSProp,AdamW-Adam
```

### Workflow 3: Test Functions (Quick Convergence Check)

```bash
# Rosenbrock function with 10 seeds
python run_full_analysis.py \
    --config configs/rosenbrock_tuning.json \
    --seeds 1,2,3,4,5,6,7,8,9,10 \
    --compare Adam-RMSProp
```

## Interpreting Results

### T-Test Results

- **p < 0.05**: Statistically significant difference (reject null hypothesis)
- **p ≥ 0.05**: No significant difference (fail to reject null)

### Effect Size (Cohen's d)

- **|d| < 0.2**: Negligible effect
- **0.2 ≤ |d| < 0.5**: Small effect
- **0.5 ≤ |d| < 0.8**: Medium effect
- **|d| ≥ 0.8**: Large effect

### Confidence Intervals

- **95% CI**: We are 95% confident the true mean lies in this range
- **Non-overlapping CIs**: Strong evidence of difference
- **Overlapping CIs**: Difference may not be significant

## Best Practices

1. **Always use ≥5 seeds** for statistical reliability
2. **Report full statistics**: mean, std, min, max, n
3. **Include error bars** on all plots
4. **Perform statistical tests** before claiming superiority
5. **Check effect sizes** - statistical significance ≠ practical significance
6. **Run unit tests** before experiments: `pytest tests/`

## Troubleshooting

**Issue**: "No module named scipy"
```bash
pip install scipy
```

**Issue**: "No results found"
- Check that experiments ran successfully
- Verify results are in `results/` directory
- Check filename patterns match expected format

**Issue**: "Tests failing"
```bash
# Re-run tests with verbose output
pytest tests/ -v --tb=short

# Check specific test
pytest tests/test_gradients.py::TestRosenbrockGradients -v
```

## Next Steps

After running multi-seed experiments:

1.  **Verify statistical significance** with t-tests
2.  **Check effect sizes** (practical significance)
3.  **Visualize with error bars** (confidence bands)
4.  **Document limitations** (sample size, assumptions)
5.  **Compare to baselines** (PyTorch defaults, published papers)

## References

- **Statistical Testing**: Welch's t-test for unequal variances
- **Effect Size**: Cohen, J. (1988). Statistical Power Analysis
- **Confidence Intervals**: 95% CI using t-distribution
- **Multiple Comparisons**: Consider Bonferroni correction if comparing >2 optimizers
