# GDSearch Experiment Results

## Directory Structure

```
results/
├── experiments/              # Raw per-run experiment data
│   ├── mnist/               # MNIST CSVs (per seed, per optimizer)
│   ├── cifar10/             # CIFAR-10 CSVs
│   ├── nlp/                 # NLP sentiment CSVs
│   ├── medical/             # Medical segmentation CSVs
│   ├── resnet/              # ResNet18 CSVs
│   └── highdim/             # High-dimensional optimization CSVs
├── analysis/                 # Post-experiment analysis
│   ├── 00_basic_statistics.csv          # Mean/std/min/max per optimizer
│   ├── 01_convergence_rates.csv         # Convergence analysis
│   └── 02_statistical_comparison.csv    # t-tests, effect sizes, p-values
├── visualizations/           # Plots and charts
│   ├── interactive/         # HTML plots (Plotly) - open in browser
│   └── static/              # PNG/PDF static plots
└── reports/                  # Markdown summaries
    └── 00_EXPERIMENT_SUMMARY.md         # Comprehensive experiment report
```

## File Naming Conventions

### Experiment CSVs
- Format: `{DATASET}_{MODEL}_{OPTIMIZER}_seed{SEED}.csv`
- Example: `MNIST_MLP_Adam_seed42.csv`

### Analysis Files
- Numbered prefix for logical ordering (00, 01, 02...)
- Descriptive names indicating content

### Interactive Plots
- Format: `{dataset}_optimizer_comparison.html`
- Open in web browser for interactive pan/zoom/hover

## Quick Access

```bash
# View convergence analysis
cat analysis/01_convergence_rates.csv | column -t -s,

# View statistical comparisons
cat analysis/02_statistical_comparison.csv | column -t -s,

# Open interactive plots
open visualizations/interactive/*.html  # macOS
xdg-open visualizations/interactive/*.html  # Linux

# Read summary report
cat reports/00_EXPERIMENT_SUMMARY.md
```

## Using Results in Python

```python
import pandas as pd
from pathlib import Path

results = Path('results')

# Load convergence data
conv = pd.read_csv(results / 'analysis/01_convergence_rates.csv')
print(conv.groupby('optimizer')['convergence_rate'].mean())

# Load statistical tests
stats = pd.read_csv(results / 'analysis/02_statistical_comparison.csv')
significant = stats[stats['is_significant']]
print(significant[['optimizer_1', 'optimizer_2', 'mean_diff', 'p_value']])

# Load specific experiment
mnist_adam = pd.read_csv(results / 'experiments/mnist/MNIST_MLP_Adam_seed42.csv')
print(mnist_adam[['epoch', 'test_acc']].tail())
```
