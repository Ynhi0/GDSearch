# 📊 Automatic Visualization Guide

**Every experiment now automatically generates comprehensive visualizations!**

## What Gets Created

### For EACH Experiment (MNIST, CIFAR10, NLP, ResNet, HighDim, etc.)

#### 🖼️ **Static Plots** (PNG, 300 DPI)
Located in: `results/visualizations/static/{experiment_name}/`

1. **Training Loss Curves** - `{experiment}_train_loss.png`
   - Loss progression over epochs
   - Mean ± standard deviation across seeds
   - Separate line for each optimizer

2. **Test Accuracy Curves** - `{experiment}_test_accuracy.png`
   - Accuracy progression over epochs
   - Mean ± standard deviation across seeds
   - Shows which optimizer learns fastest

3. **Final Performance Comparison** - `{experiment}_final_comparison.png`
   - Bar chart of final test accuracy
   - Error bars showing variability
   - Easy at-a-glance winner identification

#### 🌐 **Interactive HTML Plots**
Located in: `results/visualizations/interactive/`

- **Multi-Metric Comparison** - `{experiment}_interactive_comparison.html`
  - Interactive subplots for train_loss, test_loss, test_acc
  - Hover to see exact values
  - Pan, zoom, toggle optimizers on/off
  - Uncertainty bands for multi-seed runs
  - Export to PNG directly from browser

## Directory Structure

```
results/
├── experiments/                     # Raw CSV data
│   ├── mnist/
│   │   ├── MNIST_MLP_SGD_seed1.csv
│   │   ├── MNIST_MLP_Adam_seed1.csv
│   │   └── ...
│   ├── cifar10/
│   ├── nlp/
│   └── ...
└── visualizations/                  # ALL PLOTS HERE
    ├── static/                      # PNG files (for papers/reports)
    │   ├── mnist/
    │   │   ├── mnist_train_loss.png
    │   │   ├── mnist_test_accuracy.png
    │   │   └── mnist_final_comparison.png
    │   ├── cifar10/
    │   │   ├── cifar10_train_loss.png
    │   │   ├── cifar10_test_accuracy.png
    │   │   └── cifar10_final_comparison.png
    │   └── ...
    └── interactive/                 # HTML files (for exploration)
        ├── mnist_interactive_comparison.html
        ├── cifar10_interactive_comparison.html
        └── ...
```

## When Are Visualizations Created?

**Automatically, immediately after each experiment completes!**

```bash
python run_all_kaggle.py --experiments mnist --seeds 1,2,3
# Experiment runs...
# ✓ MNIST_MLP_SGD_seed1.csv saved
# ✓ MNIST_MLP_Adam_seed1.csv saved
# ✓ MNIST_MLP_AdamW_seed1.csv saved
# ...
# 📊 Creating visualizations for MNIST...
#    ✓ Created mnist_train_loss.png
#    ✓ Created mnist_test_accuracy.png
#    ✓ Created mnist_final_comparison.png
#    ✓ Created mnist_interactive_comparison.html
#    ✓ MNIST visualizations complete
```

No separate visualization script needed! 🎉

## Viewing Results

### Static PNG Plots
Perfect for papers, presentations, and reports.

```bash
# View all MNIST plots
open results/visualizations/static/mnist/*.png

# Or specific plot
open results/visualizations/static/mnist/mnist_test_accuracy.png
```

### Interactive HTML Plots
Perfect for exploration and analysis.

```bash
# Open in browser
open results/visualizations/interactive/mnist_interactive_comparison.html

# Or open all experiments
open results/visualizations/interactive/*.html
```

**Interactive features:**
- **Hover**: See exact values
- **Click legend**: Toggle optimizer visibility
- **Drag**: Pan the plot
- **Scroll**: Zoom in/out
- **Double-click**: Reset view
- **Camera icon**: Download as PNG

## Multi-Seed Visualization

When you run with multiple seeds (`--seeds 1,2,3`), plots automatically show:

- **Mean line**: Average across all seeds
- **Shaded area**: ±1 standard deviation
- **Individual lines**: (In interactive plots, toggle visibility)

Example:
```bash
python run_all_kaggle.py --experiments mnist --seeds 1,2,3,4,5
# Creates plots with uncertainty bands
```

## Supported Experiment Types

✅ **MNIST** - Simple MLP classification  
✅ **CIFAR10** - ResNet18 image classification  
✅ **NLP** - DistilBERT sentiment analysis  
✅ **Medical** - U-Net segmentation  
✅ **ResNet18** - Deep network training  
✅ **HighDim** - High-dimensional optimization  
✅ **2D Optimization** - Test function visualization  
✅ **Robustness** - Initial condition analysis  
✅ **SAM Sensitivity** - Hyperparameter analysis  
✅ **Ablation** - Component comparison  

## Customization

### Disable Visualizations
Not currently supported (they're generated automatically). But you can delete the `visualizations/` directory if needed.

### Generate for Existing Results
Visualizations are created from CSVs. If you have old CSVs:

```python
from run_all_kaggle import create_experiment_visualizations
from pathlib import Path

csv_files = list(Path('results/experiments/mnist').glob('*.csv'))
create_experiment_visualizations('MNIST', 'results', csv_files)
```

### Add Custom Plots
Edit `create_experiment_visualizations()` in `run_all_kaggle.py` to add your own plot types.

## Technical Details

### What Data is Plotted?

**Static plots** read from CSV columns:
- `epoch` - X-axis for all plots
- `train_loss` - Training loss curve
- `test_loss` - Test loss curve
- `test_acc` or `test_accuracy` - Accuracy curve
- `optimizer` - For grouping/coloring
- `seed` - For computing mean ± std

**Interactive plots** include all metrics in subplots:
- Up to 4 metrics per figure
- Adaptive layout (1x1, 2x1, 2x2)
- Synchronized hover across subplots

### Dependencies
- **matplotlib** - Static plots (always available)
- **plotly** - Interactive plots (installed in requirements.txt)
- **pandas** - Data loading (always available)
- **numpy** - Statistics (always available)

## Examples

### Quick Run with Visualizations
```bash
python run_all_kaggle.py --quick --experiments mnist
# Runs fast experiment + creates all plots
```

### Full Pipeline
```bash
python run_all_kaggle.py --experiments mnist,cifar10,nlp --seeds 1,2,3,4,5
# Creates visualizations for all 3 experiments
```

### Check What Was Created
```bash
# List all PNG plots
find results/visualizations/static -name "*.png"

# List all HTML plots
find results/visualizations/interactive -name "*.html"

# Count total plots
find results/visualizations -type f | wc -l
```

## Benefits

### ✅ **Zero Extra Work**
- Automatic generation after each experiment
- No separate plotting script
- No manual file management

### ✅ **Consistent Format**
- Same style across all experiments
- Same file naming convention
- Same directory structure

### ✅ **Publication Ready**
- 300 DPI PNG files for papers
- Professional matplotlib styling
- Error bars and uncertainty bands

### ✅ **Interactive Exploration**
- Zoom into specific epochs
- Toggle optimizers for comparison
- Hover for exact numbers
- Export custom views

### ✅ **Multi-Seed Support**
- Automatic mean ± std computation
- Shaded uncertainty regions
- Statistical rigor built-in

## Troubleshooting

### "No visualizations created"
**Cause**: No CSV files found  
**Fix**: Ensure experiment completed successfully

### "Interactive plots missing"
**Cause**: Plotly not installed  
**Fix**: `pip install plotly`

### "Plots look wrong"
**Cause**: CSV format mismatch  
**Fix**: Ensure CSVs have `epoch`, `optimizer`, metric columns

### "Too many plots"
**Cause**: Multiple experiments ran  
**Fix**: This is expected! Each experiment gets its own set of plots

## Summary

🎯 **Every raw CSV automatically gets:**
- 3 static PNG plots (train_loss, test_accuracy, final_comparison)
- 1 interactive HTML plot (multi-metric comparison)

📁 **All organized in:**
- `results/visualizations/static/{experiment}/` - PNG files
- `results/visualizations/interactive/` - HTML files

🚀 **No extra steps needed!**
Just run your experiment and visualizations appear automatically!

---

**Last Updated**: December 2025  
**Feature Status**: ✅ Production Ready  
**Auto-Generated**: Yes, for all experiments
