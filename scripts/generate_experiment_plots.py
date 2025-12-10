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
        
        # Extract optimizer name
        # Format: NN_{dataset}_{optimizer}_lr{lr}_seed{seed}.csv
        parts = basename.replace('.csv', '').split('_')
        
        # Find optimizer (between dataset and lr)
        optimizer = None
        for i, part in enumerate(parts):
            if part.startswith('lr'):
                # Optimizer is parts before lr
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
            if 'train_loss' in df.columns and 'epoch' in df.columns:
                ax.plot(df['epoch'], df['train_loss'], color=color, alpha=0.3, linewidth=1)
        
        # Mean line
        if dfs and 'train_loss' in dfs[0].columns:
            epochs = dfs[0]['epoch'].values
            losses = np.array([df['train_loss'].values for df in dfs if 'train_loss' in df.columns])
            if len(losses) > 0:
                mean_loss = losses.mean(axis=0)
                ax.plot(epochs, mean_loss, color=color, linewidth=2.5, label=optimizer)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Test/Validation Accuracy
    ax = axes[0, 1]
    for i, (optimizer, dfs) in enumerate(sorted(results.items())):
        color = colors[i % len(colors)]
        for df in dfs:
            acc_col = next((col for col in df.columns if 'acc' in col.lower() and 'test' in col.lower()), None)
            if acc_col and 'epoch' in df.columns:
                # Convert to percentage if needed
                acc_values = df[acc_col].values
                if acc_values.max() <= 1.0:
                    acc_values = acc_values * 100
                ax.plot(df['epoch'], acc_values, color=color, alpha=0.3, linewidth=1)
        
        # Mean line
        if dfs:
            acc_col = next((col for col in dfs[0].columns if 'acc' in col.lower() and 'test' in col.lower()), None)
            if acc_col:
                epochs = dfs[0]['epoch'].values
                accs = []
                for df in dfs:
                    if acc_col in df.columns:
                        acc_values = df[acc_col].values
                        if acc_values.max() <= 1.0:
                            acc_values = acc_values * 100
                        accs.append(acc_values)
                if accs:
                    mean_acc = np.array(accs).mean(axis=0)
                    ax.plot(epochs, mean_acc, color=color, linewidth=2.5, label=optimizer)
    
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
