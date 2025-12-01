#!/usr/bin/env python3
"""
Integrate All Unused Features into run_all_kaggle.py

This script demonstrates how to use all existing src/ modules:
1. Loss landscape visualization (probe_loss_2d)
2. Interactive plots (plot_trajectory_interactive, plot_multi_optimizer_comparison)
3. Convergence analysis (ConvergenceAnalyzer)
4. Sensitivity analysis (run_sensitivity_experiment)
5. Baseline comparison (compare optimizers)
6. Statistical analysis (compare_multiple_optimizers)

Run this AFTER running experiments to generate comprehensive analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print(" "*20 + "INTEGRATING ALL FEATURES")
print("="*80)

# ============================================================================
# 1. LOSS LANDSCAPE VISUALIZATION
# ============================================================================
print("\n1️⃣  Loss Landscape Visualization Module")
print("-"*80)
try:
    from src.visualization import loss_landscape
    print("✅ Module: src.visualization.loss_landscape")
    print("   Functions:")
    print("   - probe_loss_1d: 1D loss profile along a direction")
    print("   - probe_loss_2d: 2D loss landscape (meshgrid)")
    print("   - evaluate_loss: Compute model loss on dataloader")
    print("   Usage: After training, call probe_loss_2d to visualize loss surface")
except ImportError as e:
    print(f"❌ Could not import: {e}")

# ============================================================================
# 2. INTERACTIVE PLOTS
# ============================================================================
print("\n2️⃣  Interactive Plots Module")
print("-"*80)
try:
    from src.visualization import interactive_plots
    print("✅ Module: src.visualization.interactive_plots")
    print("   Functions:")
    print("   - plot_trajectory_interactive: 2D trajectories with contours")
    print("   - plot_loss_landscape_3d: 3D surface plot")
    print("   - animate_convergence: Animated optimization")
    print("   - plot_multi_optimizer_comparison: Multi-panel comparison")
    print("   Usage: Load CSV results and create interactive HTML plots")
except ImportError as e:
    print(f"❌ Could not import: {e}")

# ============================================================================
# 3. CONVERGENCE ANALYSIS
# ============================================================================
print("\n3️⃣  Convergence Analysis Module")
print("-"*80)
try:
    from src.experiments.convergence_analysis import ConvergenceAnalyzer
    print("✅ Module: src.experiments.convergence_analysis")
    print("   Classes:")
    print("   - ConvergenceAnalyzer: Analyze convergence rates")
    print("   Methods:")
    print("   - analyze: Single optimizer convergence")
    print("   - compare_optimizers: Compare multiple optimizers")
    print("   Detects: sublinear, linear, superlinear, stagnation")
    print("   Usage: Analyze loss trajectories from experiments")
except ImportError as e:
    print(f"❌ Could not import: {e}")

# ============================================================================
# 4. SENSITIVITY ANALYSIS
# ============================================================================
print("\n4️⃣  Sensitivity Analysis Module")
print("-"*80)
try:
    from src.analysis import sensitivity_analysis
    print("✅ Module: src.analysis.sensitivity_analysis")
    print("   Functions:")
    print("   - generate_sensitivity_grid: Create parameter grid")
    print("   - run_sensitivity_experiment: Test parameter robustness")
    print("   Usage: Test hyperparameter stability around 'best' values")
except ImportError as e:
    print(f"❌ Could not import: {e}")

# ============================================================================
# 5. BASELINE COMPARISON
# ============================================================================
print("\n5️⃣  Baseline Comparison Module")
print("-"*80)
try:
    from src.analysis import baseline_comparison
    print("✅ Module: src.analysis.baseline_comparison")
    print("   Purpose: Verify custom optimizers match PyTorch equivalents")
    print("   Validates:")
    print("   - SGD vs torch.optim.SGD")
    print("   - Adam vs torch.optim.Adam")
    print("   - Momentum implementations")
    print("   Usage: Numerical validation of optimizer correctness")
except ImportError as e:
    print(f"❌ Could not import: {e}")

# ============================================================================
# 6. ABLATION STUDY
# ============================================================================
print("\n6️⃣  Ablation Study Module")
print("-"*80)
try:
    from src.analysis import ablation_study
    print("✅ Module: src.analysis.ablation_study")
    print("   Purpose: Isolate component effects")
    print("   Components tested:")
    print("   - Momentum vs no momentum")
    print("   - Adaptive LR vs fixed LR")
    print("   - Weight decay effects")
    print("   Usage: Understand which components matter most")
except ImportError as e:
    print(f"❌ Could not import: {e}")

# ============================================================================
# 7. STATISTICAL ANALYSIS
# ============================================================================
print("\n7️⃣  Statistical Analysis Module")
print("-"*80)
try:
    from src.analysis import statistical_analysis
    print("✅ Module: src.analysis.statistical_analysis")
    print("   Functions:")
    print("   - compare_two_optimizers: t-test, Cohen's d, confidence intervals")
    print("   - compare_multiple_optimizers: ANOVA, pairwise tests")
    print("   - power_analysis_report: Statistical power calculation")
    print("   Usage: Rigorous statistical validation of results")
except ImportError as e:
    print(f"❌ Could not import: {e}")

# ============================================================================
# DEMO: How to use these modules
# ============================================================================
print("\n" + "="*80)
print(" "*20 + "USAGE EXAMPLES")
print("="*80)

print("\n📖 Example 1: Convergence Analysis")
print("-"*80)
print("""
from src.experiments.convergence_analysis import ConvergenceAnalyzer

# Load experiment results
df = pd.read_csv('results/NN_MLP_MNIST_Adam_lr0.001_seed42.csv')

# Analyze convergence
analyzer = ConvergenceAnalyzer(tolerance=1e-4, window_size=20)
metrics = analyzer.analyze(df['test_loss'].values)

print(f"Convergence rate: {metrics['convergence_rate']}")
print(f"Converged at epoch: {metrics['convergence_epoch']}")
""")

print("\n📖 Example 2: Interactive Plots")
print("-"*80)
print("""
from src.visualization.interactive_plots import plot_multi_optimizer_comparison

# Load results from multiple optimizers
df = pd.concat([
    pd.read_csv('results/NN_MLP_MNIST_SGD_lr0.01_seed42.csv'),
    pd.read_csv('results/NN_MLP_MNIST_Adam_lr0.001_seed42.csv'),
])

# Create interactive comparison
fig = plot_multi_optimizer_comparison(
    df,
    optimizer_col='optimizer',
    epoch_col='epoch',
    metric_cols=['train_loss', 'test_loss', 'test_acc']
)
fig.write_html('optimizer_comparison.html')
""")

print("\n📖 Example 3: Loss Landscape")
print("-"*80)
print("""
import torch
from src.visualization.loss_landscape import probe_loss_2d

# After training a model
model = ... # trained model
loader = ... # DataLoader
criterion = torch.nn.CrossEntropyLoss()

# Generate random orthogonal directions
dir1 = torch.randn(sum(p.numel() for p in model.parameters()))
dir1 = dir1 / dir1.norm()
dir2 = torch.randn_like(dir1)
dir2 = dir2 - (dir1 @ dir2) * dir1  # orthogonalize
dir2 = dir2 / dir2.norm()

# Probe 2D landscape
alphas = np.linspace(-1, 1, 25)
betas = np.linspace(-1, 1, 25)
A, B, Z = probe_loss_2d(model, loader, criterion, device, dir1, dir2, alphas, betas)

# Plot
import matplotlib.pyplot as plt
plt.contourf(A, B, Z, levels=20)
plt.colorbar(label='Loss')
plt.title('Loss Landscape')
plt.savefig('loss_landscape.png')
""")

print("\n📖 Example 4: Sensitivity Analysis")
print("-"*80)
print("""
from src.analysis.sensitivity_analysis import run_sensitivity_experiment

base_config = {
    'model': 'MLP',
    'dataset': 'MNIST',
    'optimizer': 'Adam',
    'lr': 0.001,  # <- test this parameter
    'epochs': 10,
    'batch_size': 128
}

# Test learning rate sensitivity
lr_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]
results_df = run_sensitivity_experiment(
    base_config,
    param_name='lr',
    param_values=lr_values
)

print(results_df)
""")

print("\n📖 Example 5: Statistical Comparison")
print("-"*80)
print("""
from src.analysis.statistical_analysis import compare_two_optimizers

# Load results from multi-seed experiments
sgd_results = [0.95, 0.94, 0.96, 0.95, 0.94]  # test accuracies
adam_results = [0.97, 0.96, 0.97, 0.98, 0.96]

stats = compare_two_optimizers(
    sgd_results,
    adam_results,
    opt1_name='SGD',
    opt2_name='Adam',
    alpha=0.05
)

print(f"Mean difference: {stats['mean_diff']:.4f}")
print(f"p-value: {stats['p_value']:.4e}")
print(f"Cohen's d: {stats['cohens_d']:.3f}")
print(f"Significant: {stats['is_significant']}")
""")

# ============================================================================
# CHECK RESULTS DIRECTORY
# ============================================================================
print("\n" + "="*80)
print(" "*20 + "CHECKING AVAILABLE DATA")
print("="*80)

results_dir = Path('results')
if results_dir.exists():
    csv_files = list(results_dir.glob('**/*.csv'))
    json_files = list(results_dir.glob('**/*.json'))
    
    print(f"\n📁 Found in results/:")
    print(f"   - {len(csv_files)} CSV files")
    print(f"   - {len(json_files)} JSON files")
    
    if csv_files:
        print("\n📊 Sample CSV files:")
        for f in csv_files[:5]:
            print(f"   - {f.name}")
        if len(csv_files) > 5:
            print(f"   ... and {len(csv_files) - 5} more")
    
    # Try to load and analyze one
    if csv_files:
        print("\n🔍 Quick Analysis of First CSV:")
        try:
            df = pd.read_csv(csv_files[0])
            print(f"   File: {csv_files[0].name}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            if 'test_loss' in df.columns and len(df) > 10:
                from src.experiments.convergence_analysis import ConvergenceAnalyzer
                analyzer = ConvergenceAnalyzer(tolerance=1e-4, window_size=10)
                metrics = analyzer.analyze(df['test_loss'].values)
                print(f"\n   Convergence Analysis:")
                print(f"   - Rate: {metrics['convergence_rate']}")
                print(f"   - Epoch: {metrics['convergence_epoch']}")
                print(f"   - Final loss: {metrics['final_loss']:.6f}")
        except Exception as e:
            print(f"   Could not analyze: {e}")
else:
    print("\n⚠️  No results/ directory found")
    print("   Run experiments first: python run_all_kaggle.py")

# ============================================================================
# INTEGRATION RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print(" "*20 + "INTEGRATION RECOMMENDATIONS")
print("="*80)

recommendations = [
    ("After training", "Add loss landscape visualization", "probe_loss_2d()"),
    ("After experiments", "Generate interactive plots", "plot_multi_optimizer_comparison()"),
    ("For all runs", "Track convergence", "ConvergenceAnalyzer.analyze()"),
    ("Before publishing", "Run statistical tests", "compare_multiple_optimizers()"),
    ("For tuning", "Test sensitivity", "run_sensitivity_experiment()"),
    ("For validation", "Verify baselines", "baseline_comparison.py"),
]

for i, (when, what, how) in enumerate(recommendations, 1):
    print(f"\n{i}. {when}:")
    print(f"   → {what}")
    print(f"   → Use: {how}")

print("\n" + "="*80)
print("✅ FEATURE INTEGRATION GUIDE COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Run this script: python integrate_all_features.py")
print("2. Review the usage examples above")
print("3. Add these modules to your experiment pipeline")
print("4. Generate comprehensive analysis outputs")
print("\nAll src/ modules are professional and ready to use!")
print("="*80)
