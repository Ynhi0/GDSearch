#!/usr/bin/env python3
"""
Universal visualization generator for all experiments.
Reads CSV files from results/ and automatically generates high-quality plots.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def _get_epochs_and_test_acc(df):
    """Return (epochs, test_accuracy_series_in_percent_or_None).

    Handles missing `epoch` column, accepts either `test_acc` or `test_accuracy` (or variants).
    Converts values in [0,1] to percentages.
    """
    # Epochs
    epoch_col = next((col for col in df.columns if col.strip().lower() == 'epoch' or 'epoch' in col.lower()), None)
    if epoch_col is not None:
        epochs = df[epoch_col].values
    else:
        epochs = np.arange(1, len(df) + 1)

    # Test accuracy column detection
    acc_col = next((col for col in df.columns if 'test' in col.lower() and ('acc' in col.lower() or 'accuracy' in col.lower())), None)
    if acc_col is None:
        return epochs, None

    acc_vals = pd.to_numeric(df[acc_col], errors='coerce')
    if acc_vals.isna().all():
        return epochs, None

    # Convert to percentage when in [0,1]
    if acc_vals.max() <= 1.01:
        acc_vals = acc_vals * 100.0

    return epochs, acc_vals.values


def plot_training_curves(csv_files: List[str], output_dir: Path, title: str = "Training Curves"):
    """
    Generate training curves from CSV files.
    Handles MNIST, CIFAR-10, NLP, and other NN experiments.
    """
    if not csv_files:
        return

    # Group by optimizer
    results = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        basename = os.path.basename(csv_file)

        # Extract optimizer name robustly
        # Strategy: search for known optimizer tokens in the filename (handles hyphens and variants)
        norm_name = basename.replace('.csv', '').replace('-', '').replace('+', '_').upper()
        known_opts = [
            'SGD_MOMENTUM', 'SGDMOMENTUM', 'ADAMW', 'ADAM', 'RMSPROP', 'AMSGRAD',
            'SAM_SGD', 'SAM_ADAM', 'LOOKAHEAD_SGD', 'LOOKAHEAD_ADAM', 'ADABOUND', 'RADAM', 'LAMB', 'SGD', 'SAM'
        ]
        # Sort by length descending to match longest patterns first (e.g., SGD_MOMENTUM before SGD)
        known_opts.sort(key=len, reverse=True)
        
        optimizer = None
        for opt in known_opts:
            if opt.upper() in norm_name:
                optimizer = opt
                break

        # Fallback to original position-based extraction if not found
        if not optimizer:
            parts = basename.replace('.csv', '').split('_')
            for i, part in enumerate(parts):
                if part.startswith('lr'):
                    dataset_idx = next((j for j, p in enumerate(parts) if p.upper() in ['MNIST', 'CIFAR10', 'IMDB', 'MEDICAL', 'SIMPLECIFAR10']), 0)
                    optimizer = '_'.join(parts[dataset_idx+1:i])
                    break

        if not optimizer:
            optimizer = 'Unknown'

        if optimizer not in results:
            results[optimizer] = []
        results[optimizer].append(df)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FFD93D', '#B8E6F1']

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for i, (optimizer, dfs) in enumerate(sorted(results.items())):
        color = colors[i % len(colors)]
        for df in dfs:
            x_col = 'epoch' if 'epoch' in df.columns else ('iteration' if 'iteration' in df.columns else None)
            y_col = 'train_loss' if 'train_loss' in df.columns else ('loss' if 'loss' in df.columns else None)
            if x_col and y_col:
                ax.plot(df[x_col], df[y_col], color=color, alpha=0.3, linewidth=1)

        # Mean line
        if dfs:
            valid_dfs = []
            x_col_mean = None
            y_col_mean = None
            for df in dfs:
                x_col = 'epoch' if 'epoch' in df.columns else ('iteration' if 'iteration' in df.columns else None)
                y_col = 'train_loss' if 'train_loss' in df.columns else ('loss' if 'loss' in df.columns else None)
                if x_col and y_col:
                    if not x_col_mean: x_col_mean, y_col_mean = x_col, y_col
                    valid_dfs.append(df)

            if valid_dfs and x_col_mean and y_col_mean:
                # Handle varying lengths (e.g. 2D might stop early)
                max_len = max(len(d) for d in valid_dfs)
                # Ensure all are interpolated to same length or just average up to min_len
                min_len = min(len(d) for d in valid_dfs)
                losses = np.array([d[y_col_mean].values[:min_len] for d in valid_dfs])
                epochs = valid_dfs[0][x_col_mean].values[:min_len]
                if len(losses) > 0:
                    mean_loss = losses.mean(axis=0)
                    ax.plot(epochs, mean_loss, color=color, linewidth=2.5, label=optimizer)

    ax.set_xlabel('Epoch / Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Test/Validation Accuracy
    ax = axes[0, 1]
    for i, (optimizer, dfs) in enumerate(sorted(results.items())):
        color = colors[i % len(colors)]
        # Collect per-run (epochs, acc) and plot raw runs
        runs_for_mean = []
        for df in dfs:
            epochs, acc = _get_epochs_and_test_acc(df)
            if acc is not None:
                ax.plot(epochs, acc, color=color, alpha=0.3, linewidth=1)
                runs_for_mean.append((epochs, acc))

        # Mean line (align runs by common epoch grid using interpolation)
        if runs_for_mean:
            common_epochs = np.arange(1, int(max(e.max() for e, _ in runs_for_mean)) + 1)
            aligned_accs = []
            for e, a in runs_for_mean:
                s = pd.Series(a, index=e)
                s = s.reindex(common_epochs).interpolate().ffill().bfill().values
                aligned_accs.append(s)
            mean_acc = np.mean(np.vstack(aligned_accs), axis=0)
            ax.plot(common_epochs, mean_acc, color=color, linewidth=2.5, label=optimizer)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Final Performance Bar Chart
    ax = axes[1, 0]
    final_metrics = {}
    final_stds = {}

    for optimizer, dfs in results.items():
        # Get final test accuracy from each run
        final_vals = []
        for df in dfs:
            acc_col = next((col for col in df.columns if 'acc' in col.lower() and 'test' in col.lower()), None)
            if acc_col:
                val = df[acc_col].iloc[-1]
                if val <= 1.0:
                    val = val * 100
                final_vals.append(val)

        if final_vals:
            final_metrics[optimizer] = np.mean(final_vals)
            final_stds[optimizer] = np.std(final_vals) if len(final_vals) > 1 else 0

    if final_metrics:
        optimizers_sorted = sorted(final_metrics.keys(), key=lambda x: final_metrics[x], reverse=True)
        x_pos = np.arange(len(optimizers_sorted))
        bars = ax.bar(x_pos, [final_metrics[opt] for opt in optimizers_sorted],
                      yerr=[final_stds[opt] for opt in optimizers_sorted],
                      color=[colors[i % len(colors)] for i in range(len(optimizers_sorted))],
                      alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizers_sorted, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Final Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Final Performance', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for bar, opt in zip(bars, optimizers_sorted):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{final_metrics[opt]:.2f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Plot 4: Training speed comparison
    ax = axes[1, 1]
    speeds = {}
    for optimizer, dfs in results.items():
        run_speeds = []
        for df in dfs:
            if 'elapsed_seconds' in df.columns and df['elapsed_seconds'].iloc[0] > 0:
                # Rough estimate: assume 50k samples for MNIST/CIFAR
                total_epochs = df['epoch'].max()
                elapsed = df['elapsed_seconds'].iloc[0]
                samples_per_sec = (50000 * total_epochs) / elapsed
                run_speeds.append(samples_per_sec)
        if run_speeds:
            speeds[optimizer] = np.mean(run_speeds)

    if speeds:
        optimizers_sorted = sorted(speeds.keys(), key=lambda x: speeds[x], reverse=True)
        x_pos = np.arange(len(optimizers_sorted))
        ax.bar(x_pos, [speeds[opt] for opt in optimizers_sorted],
               color=[colors[i % len(colors)] for i in range(len(optimizers_sorted))],
               alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizers_sorted, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Training Speed (samples/sec)', fontsize=12, fontweight='bold')
        ax.set_title('Training Efficiency', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No timing data available',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)

    plt.tight_layout()
    output_file = output_dir / f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def generate_all_plots(results_dir: str = 'results'):
    """
    Automatically generate plots for all experiments in results directory.
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    print(f"🎨 Generating visualizations from: {results_dir}")
    print("="*80)

    # Find all CSV files
    try:
        all_csvs = glob.glob(str(results_path / "**/*.csv"), recursive=True)
    except Exception as e:
        print(f"⚠️  Error finding CSV files: {e}")
        all_csvs = []

    if not all_csvs:
        print("⚠️  No CSV files found")
        return

    print(f"Found {len(all_csvs)} CSV files")

    # Group by experiment type
    experiments = {
        'MNIST': [],
        'CIFAR10': [],
        'IMDB': [],
        'Medical': [],
        '2D': [],
        'Other': []
    }

    for csv_file in all_csvs:
        basename = os.path.basename(csv_file).upper()
        categorized = False
        for exp_type in experiments.keys():
            if exp_type.upper() in basename:
                experiments[exp_type].append(csv_file)
                categorized = True
                break
        if not categorized:
            experiments['Other'].append(csv_file)

    # Generate plots for each category
    viz_dir = results_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    plots_created = 0
    for exp_type, csv_files in experiments.items():
        if csv_files:
            print(f"\n📊 {exp_type}: {len(csv_files)} files")
            try:
                plot_training_curves(csv_files, viz_dir, title=f"{exp_type} Training Results")
                plots_created += 1
            except Exception as e:
                print(f"   ⚠️  Error: {e}")

    print("\n" + "="*80)
    print(f"✅ Created {plots_created} visualization sets in: {viz_dir}")
    print(f"   All plots are high-quality (300 DPI)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate high-quality plots from experiment CSVs')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory containing CSV files')

    args = parser.parse_args()

    generate_all_plots(args.results_dir)
