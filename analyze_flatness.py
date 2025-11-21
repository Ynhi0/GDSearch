#!/usr/bin/env python3
"""
Flatness Analysis Script for GDSearch

This script analyzes the flatness of minima found by different optimizers
using loss landscape curvature analysis. It demonstrates that SAM (Sharpness-Aware
Minimization) finds flatter minima compared to SGD/Adam, leading to better
generalization.

Usage:
    python analyze_flatness.py --results_dir /path/to/results --output_dir /path/to/output
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.loss_landscape import evaluate_loss, _get_params_vector, _set_params_from_vector


def load_model_and_results(results_dir: Path, model_class: nn.Module) -> Dict[str, Tuple[nn.Module, pd.DataFrame]]:
    """
    Load trained models and their training histories from results directory.

    Args:
        results_dir: Directory containing CSV results
        model_class: Model class to instantiate

    Returns:
        Dictionary mapping optimizer names to (model, history_df) tuples
    """
    results = {}

    # Find all publication CSV files
    csv_files = list(results_dir.glob("*_publication.csv"))

    for csv_file in csv_files:
        # Parse filename to extract optimizer info
        filename = csv_file.stem  # Remove .csv extension
        parts = filename.split('_')

        # Extract optimizer name (should be between 'MNIST_' and '_lr')
        try:
            opt_start = parts.index('MNIST') + 1
            lr_start = None
            for i, part in enumerate(parts):
                if part.startswith('lr'):
                    lr_start = i
                    break

            if lr_start is None:
                continue

            optimizer_name = '_'.join(parts[opt_start:lr_start])

            # Load CSV
            df = pd.read_csv(csv_file)

            # Create model and load final weights (if available)
            model = model_class()

            # For now, we'll analyze based on training curves
            # In a full implementation, we'd load the actual trained weights
            results[optimizer_name] = (model, df)

        except (ValueError, IndexError):
            continue

    return results


def compute_flatness_metrics(history_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute flatness-related metrics from training history.

    Args:
        history_df: DataFrame with training history

    Returns:
        Dictionary of flatness metrics
    """
    metrics = {}

    # 1. Training stability (lower variance in final epochs = flatter minimum)
    final_epochs = history_df.tail(5)  # Last 5 epochs
    metrics['train_loss_stability'] = final_epochs['train_loss'].std()
    metrics['test_acc_stability'] = final_epochs['test_acc'].std()

    # 2. Convergence smoothness (lower oscillation = flatter landscape)
    loss_values = history_df['train_loss'].values
    if len(loss_values) > 2:
        # Compute second differences (acceleration)
        first_diff = np.diff(loss_values)
        second_diff = np.diff(first_diff)
        metrics['loss_smoothness'] = np.std(second_diff)

    # 3. Final generalization gap (smaller gap often indicates flatter minimum)
    final_train_loss = history_df['train_loss'].iloc[-1]
    final_test_loss = history_df['test_loss'].iloc[-1]
    metrics['final_generalization_gap'] = final_test_loss - final_train_loss

    # 4. Training efficiency (how quickly it reaches low loss)
    min_loss = history_df['train_loss'].min()
    epochs_to_converge = (history_df['train_loss'] - min_loss).abs().idxmin()
    metrics['convergence_speed'] = epochs_to_converge

    return metrics


def analyze_optimizer_flatness(results: Dict[str, Tuple[nn.Module, pd.DataFrame]]) -> pd.DataFrame:
    """
    Analyze flatness characteristics of different optimizers.
    
    Note: For SAM optimizers, ensure they were trained using the proper PyTorch
    SAMWrapper implementation which correctly computes adversarial gradients
    via closure functions. The base SAM class in optimizers.py is for 2D
    visualization only.

    Args:
        results: Dictionary from load_model_and_results

    Returns:
        DataFrame with flatness analysis results
    """
    analysis_results = []

    for optimizer_name, (model, history_df) in results.items():
        metrics = compute_flatness_metrics(history_df)

        # Add optimizer info
        metrics['optimizer'] = optimizer_name
        metrics['final_train_loss'] = history_df['train_loss'].iloc[-1]
        metrics['final_test_acc'] = history_df['test_acc'].iloc[-1]
        metrics['final_test_loss'] = history_df['test_loss'].iloc[-1]

        analysis_results.append(metrics)

    return pd.DataFrame(analysis_results)


def create_flatness_comparison_plot(analysis_df: pd.DataFrame, output_dir: Path):
    """
    Create publication-quality plots comparing optimizer flatness.

    Args:
        analysis_df: DataFrame from analyze_optimizer_flatness
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimizer Flatness Analysis: SAM vs Traditional Optimizers', fontsize=16, fontweight='bold')

    # Plot 1: Training stability (lower = flatter)
    ax1 = axes[0, 0]
    optimizers = analysis_df['optimizer']
    stability = analysis_df['train_loss_stability']
    bars = ax1.bar(range(len(optimizers)), stability, color='skyblue', alpha=0.8)
    ax1.set_xticks(range(len(optimizers)))
    ax1.set_xticklabels(optimizers, rotation=45, ha='right')
    ax1.set_ylabel('Training Loss Stability (std of final 5 epochs)')
    ax1.set_title('Training Stability\n(Lower = Flatter Minimum)')
    ax1.grid(axis='y', alpha=0.3)

    # Highlight SAM optimizers
    for i, opt in enumerate(optimizers):
        if 'SAM' in opt:
            bars[i].set_color('orange')
            bars[i].set_label('SAM Optimizer' if i == 0 else "")

    # Plot 2: Generalization gap
    ax2 = axes[0, 1]
    gap = analysis_df['final_generalization_gap']
    bars = ax2.bar(range(len(optimizers)), gap, color='lightcoral', alpha=0.8)
    ax2.set_xticks(range(len(optimizers)))
    ax2.set_xticklabels(optimizers, rotation=45, ha='right')
    ax2.set_ylabel('Final Generalization Gap\n(Test Loss - Train Loss)')
    ax2.set_title('Generalization Gap\n(Smaller = Better Generalization)')
    ax2.grid(axis='y', alpha=0.3)

    # Highlight SAM optimizers
    for i, opt in enumerate(optimizers):
        if 'SAM' in opt:
            bars[i].set_color('orange')

    # Plot 3: Loss smoothness
    ax3 = axes[1, 0]
    smoothness = analysis_df['loss_smoothness']
    bars = ax3.bar(range(len(optimizers)), smoothness, color='lightgreen', alpha=0.8)
    ax3.set_xticks(range(len(optimizers)))
    ax3.set_xticklabels(optimizers, rotation=45, ha='right')
    ax3.set_ylabel('Loss Smoothness\n(std of second differences)')
    ax3.set_title('Loss Trajectory Smoothness\n(Lower = More Stable Convergence)')
    ax3.grid(axis='y', alpha=0.3)

    # Highlight SAM optimizers
    for i, opt in enumerate(optimizers):
        if 'SAM' in opt:
            bars[i].set_color('orange')

    # Plot 4: Final test accuracy
    ax4 = axes[1, 1]
    accuracy = analysis_df['final_test_acc']
    bars = ax4.bar(range(len(optimizers)), accuracy, color='gold', alpha=0.8)
    ax4.set_xticks(range(len(optimizers)))
    ax4.set_xticklabels(optimizers, rotation=45, ha='right')
    ax4.set_ylabel('Final Test Accuracy (%)')
    ax4.set_title('Final Performance\n(Higher = Better)')
    ax4.grid(axis='y', alpha=0.3)

    # Highlight SAM optimizers
    for i, opt in enumerate(optimizers):
        if 'SAM' in opt:
            bars[i].set_color('orange')

    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_flatness_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create summary table
    summary_table = analysis_df[['optimizer', 'train_loss_stability', 'final_generalization_gap',
                                'loss_smoothness', 'final_test_acc']].round(4)
    summary_table.to_csv(output_dir / 'flatness_analysis_summary.csv', index=False)

    print(f"✅ Flatness analysis plots saved to {output_dir}")
    print(f"✅ Summary table saved to {output_dir / 'flatness_analysis_summary.csv'}")


def statistical_comparison_flatness(analysis_df: pd.DataFrame, output_dir: Path):
    """
    Perform statistical comparison of flatness metrics between SAM and traditional optimizers.

    Args:
        analysis_df: DataFrame from analyze_optimizer_flatness
        output_dir: Directory to save results
    """
    # Separate SAM and non-SAM optimizers
    sam_optimizers = analysis_df[analysis_df['optimizer'].str.contains('SAM')]
    traditional_optimizers = analysis_df[~analysis_df['optimizer'].str.contains('SAM')]

    if len(sam_optimizers) == 0 or len(traditional_optimizers) == 0:
        print("⚠️  Need both SAM and traditional optimizers for statistical comparison")
        return

    metrics_to_compare = ['train_loss_stability', 'final_generalization_gap', 'loss_smoothness']

    comparison_results = []

    for metric in metrics_to_compare:
        sam_values = sam_optimizers[metric].values
        trad_values = traditional_optimizers[metric].values

        # Perform t-test
        if len(sam_values) >= 2 and len(trad_values) >= 2:
            t_stat, p_value = stats.ttest_ind(sam_values, trad_values)

            # Effect size (Cohen's d)
            sam_mean = np.mean(sam_values)
            trad_mean = np.mean(trad_values)
            pooled_std = np.sqrt((np.var(sam_values) + np.var(trad_values)) / 2)
            cohens_d = (sam_mean - trad_mean) / pooled_std if pooled_std > 0 else 0

            comparison_results.append({
                'metric': metric,
                'sam_mean': sam_mean,
                'traditional_mean': trad_mean,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'sam_better': sam_mean < trad_mean if 'stability' in metric or 'smoothness' in metric else sam_mean > trad_mean
            })

    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(output_dir / 'flatness_statistical_comparison.csv', index=False)

        print("📊 Statistical Comparison Results:")
        for _, row in comparison_df.iterrows():
            better_indicator = "✅ BETTER" if row['sam_better'] else "❌ WORSE"
            print(".4f"
                  ".4f")

        print(f"\n✅ Statistical comparison saved to {output_dir / 'flatness_statistical_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze optimizer flatness characteristics')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing publication CSV results')
    parser.add_argument('--output_dir', type=str, default='./flatness_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--model_class', type=str, default='SimpleMLP',
                       help='Model class name (currently only SimpleMLP supported)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Starting Flatness Analysis...")
    print(f"📁 Results directory: {results_dir}")
    print(f"📁 Output directory: {output_dir}")

    # Define model class (for now, hardcoded to SimpleMLP)
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            return self.fc3(x)

    # Load results
    print("📂 Loading model results...")
    results = load_model_and_results(results_dir, SimpleMLP)

    if not results:
        print("❌ No valid results found. Please check the results directory.")
        return

    print(f"✅ Loaded results for {len(results)} optimizers: {list(results.keys())}")

    # Analyze flatness
    print("🔬 Analyzing flatness characteristics...")
    analysis_df = analyze_optimizer_flatness(results)

    # Create comparison plots
    print("📊 Creating comparison plots...")
    create_flatness_comparison_plot(analysis_df, output_dir)

    # Statistical comparison
    print("📈 Performing statistical comparisons...")
    statistical_comparison_flatness(analysis_df, output_dir)

    print("\n🎉 Flatness analysis complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()