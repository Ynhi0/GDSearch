#!/usr/bin/env python3
"""
Advanced Training Features Ablation Study

Comprehensive ablation study comparing the impact of:
1. Mixed Precision Training (AMP)
2. Label Smoothing
3. Model EMA (Exponential Moving Average)
4. Combinations of the above

Academic rigor:
- Multi-seed experiments for statistical significance
- Controlled comparisons (change ONE variable at a time)
- Measure: accuracy, loss, training time, memory usage
- Statistical tests with effect sizes
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.utils.constants import MNIST_MEAN, MNIST_STD
from src.core.training_utils import set_seed
import numpy as np
import pandas as pd
import time
import argparse
from typing import Dict, List, Tuple, Optional
import logging

from src.core.training_utils import (
    LabelSmoothingCrossEntropy,
    ModelEMA,
    AMPWrapper,
    get_loss_function,
    create_amp_wrapper,
    create_model_ema
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import centralized model (FIX #4: Remove duplicate SimpleCNN)
from src.core.models import SimpleCNN

# Removed duplicate set_seed - using from src.core.training_utils


def train_epoch(model, loader, optimizer, criterion, device,
                amp: Optional[AMPWrapper] = None,
                ema: Optional[ModelEMA] = None,
                grad_clip: float = 1.0):
    """Train for one epoch with optional AMP and EMA"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass with optional AMP
        if amp is not None:
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            amp.backward(loss, optimizer)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        if amp is not None:
            amp.step(optimizer)
            amp.update()
        else:
            optimizer.step()

        # Update EMA if enabled
        if ema is not None:
            ema.update(model)

        # Track metrics
        batch_size = targets.size(0)
        # BUG FIX: Weight loss by batch size for correct averaging
        total_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # BUG FIX: Divide by total samples, not number of batches
    return total_loss / max(1, total), 100.0 * correct / max(1, total)


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # BUG FIX: Weight loss by batch size for correct averaging
            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # BUG FIX: Divide by total samples, not number of batches
    return total_loss / max(1, total), 100.0 * correct / max(1, total)


def run_single_experiment(
    config: Dict,
    train_dataset,
    test_dataset,
    device: torch.device,
    epochs: int = 10,
    seed: int = 42,
    results_dir: Optional[str] = None
) -> Dict:
    """
    Run a single training experiment with given configuration.

    Args:
        config: Dictionary with keys:
            - use_amp: bool
            - use_label_smoothing: bool (smoothing factor if enabled)
            - use_ema: bool (decay rate if enabled)
        train_dataset: Training dataset (not loader)
        test_dataset: Test dataset (not loader)
        device: Device to use
        epochs: Number of training epochs
        seed: Random seed

    Returns:
        Dictionary with results
    """
    set_seed(seed)

    # Create DataLoaders with seed-specific RNG state
    from src.core.dataloader_utils import make_dataloader
    train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=seed, num_workers=2, pin_memory=True)
    test_loader = make_dataloader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Create model
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Setup loss function (with optional label smoothing)
    if config.get('use_label_smoothing', False):
        smoothing = float(config.get('label_smoothing_factor', 0.1))
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Setup AMP if enabled
    amp = None
    if config.get('use_amp', False) and torch.cuda.is_available():
        amp = create_amp_wrapper(enabled=True)

    # Setup EMA if enabled
    ema = None
    if config.get('use_ema', False):
        decay = float(config.get('ema_decay', 0.9999))
        ema = create_model_ema(model, decay=decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Track metrics
    history = []
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() and device.type == 'cuda' else 0

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            amp=amp, ema=ema
        )

        # Evaluate with standard model
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Evaluate with EMA model if enabled
        ema_test_acc = None
        if ema is not None:
            ema_test_loss, ema_test_acc = evaluate(ema.shadow, test_loader, criterion, device)

        scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'ema_test_acc': ema_test_acc if ema_test_acc is not None else test_acc
        })

    # Final metrics
    end_time = time.time()
    # Only query CUDA memory if device is actually a CUDA device
    if torch.cuda.is_available() and device.type == 'cuda':
        end_memory = torch.cuda.memory_allocated(device)
    else:
        end_memory = 0

    # Guard against empty history from early abort
    if not history:
        return {
            'config': config,
            'final_test_acc': float('nan'),
            'final_ema_acc': float('nan'),
            'best_test_acc': float('nan'),
            'best_ema_acc': float('nan'),
            'peak_val_acc': float('nan'),
            'peak_val_epoch': -1,
            'training_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'history': []
        }

    final_test_acc = history[-1]['test_acc']
    final_ema_acc = history[-1]['ema_test_acc']

    # NEW: Save history to CSV for per-run analysis
    if results_dir:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        history_df = pd.DataFrame(history)
        # Use sanitized filename: AdvAblation_{config_name}_seed{seed}.csv
        config_name = config.get('name', 'unknown').replace('+', '_')
        filename = f"AdvAblation_{config_name}_seed{seed}.csv"
        history_df.to_csv(results_path / filename, index=False)
        print(f"    Saved run history to {filename}")

    return {
        'config': config,
        'final_test_acc': final_test_acc,
        'final_ema_acc': final_ema_acc,
        'best_test_acc': max(h['test_acc'] for h in history),
        'best_ema_acc': max(h['ema_test_acc'] for h in history),
        'training_time': end_time - start_time,
        'memory_used': (end_memory - start_memory) / 1024**2,  # MB
        'history': history,
        'seed': seed
    }


def run_ablation_study(
    results_dir: str = "results/advanced_training_ablation",
    seeds: List[int] = [1, 2, 3, 4, 5],
    epochs: int = 10,
    quick: bool = False
):
    """
    Run comprehensive ablation study for advanced training features.

    Experimental design:
    1. Baseline (no advanced features)
    2. AMP only
    3. Label Smoothing only
    4. EMA only
    5. AMP + Label Smoothing
    6. AMP + EMA
    7. Label Smoothing + EMA
    8. All combined (AMP + Label Smoothing + EMA)

    Statistical validity:
    - Multiple seeds for each configuration
    - Controlled experiments (one variable at a time)
    - Report mean ± std for all metrics
    """
    print("="*80)
    print("🔬 ADVANCED TRAINING FEATURES ABLATION STUDY")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")

    # Setup data loaders
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform_test
    )

    if quick:
        # Use subset for quick testing
        train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
        test_dataset = torch.utils.data.Subset(test_dataset, range(1000))

    # Define ablation configurations
    configurations = [
        {
            'name': 'Baseline',
            'use_amp': False,
            'use_label_smoothing': False,
            'use_ema': False
        },
        {
            'name': 'AMP_only',
            'use_amp': True,
            'use_label_smoothing': False,
            'use_ema': False
        },
        {
            'name': 'LabelSmoothing_only',
            'use_amp': False,
            'use_label_smoothing': True,
            'label_smoothing_factor': 0.1,
            'use_ema': False
        },
        {
            'name': 'EMA_only',
            'use_amp': False,
            'use_label_smoothing': False,
            'use_ema': True,
            'ema_decay': 0.9999
        },
        {
            'name': 'AMP+LabelSmoothing',
            'use_amp': True,
            'use_label_smoothing': True,
            'label_smoothing_factor': 0.1,
            'use_ema': False
        },
        {
            'name': 'AMP+EMA',
            'use_amp': True,
            'use_label_smoothing': False,
            'use_ema': True,
            'ema_decay': 0.9999
        },
        {
            'name': 'LabelSmoothing+EMA',
            'use_amp': False,
            'use_label_smoothing': True,
            'label_smoothing_factor': 0.1,
            'use_ema': True,
            'ema_decay': 0.9999
        },
        {
            'name': 'All_Combined',
            'use_amp': True,
            'use_label_smoothing': True,
            'label_smoothing_factor': 0.1,
            'use_ema': True,
            'ema_decay': 0.9999
        }
    ]

    # Run experiments
    all_results = []

    for config in configurations:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"{'='*80}")

        config_results = []

        for seed in seeds:
            print(f"  Running seed {seed}...")
            result = run_single_experiment(
                config, train_dataset, test_dataset, device, epochs, seed,
                results_dir=results_dir
            )
            config_results.append(result)

            print(f"    Final Test Acc: {result['final_test_acc']:.2f}%")
            print(f"    Final EMA Acc: {result['final_ema_acc']:.2f}%")
            print(f"    Training Time: {result['training_time']:.2f}s")

        # Aggregate results across seeds
        test_accs = [r['final_test_acc'] for r in config_results]
        ema_accs = [r['final_ema_acc'] for r in config_results]
        times = [r['training_time'] for r in config_results]

        all_results.append({
            'configuration': config['name'],
            'use_amp': config['use_amp'],
            'use_label_smoothing': config['use_label_smoothing'],
            'use_ema': config['use_ema'],
            'mean_test_acc': np.mean(test_accs),
            'std_test_acc': np.std(test_accs),
            'test_accuracy': np.mean(test_accs),  # Add for plotting compatibility
            'mean_ema_acc': np.mean(ema_accs),
            'std_ema_acc': np.std(ema_accs),
            'mean_training_time': np.mean(times),
            'std_training_time': np.std(times),
            'training_time': np.mean(times),  # Add for plotting compatibility
            'n_seeds': len(seeds),
            'seeds': seeds
        })

        print(f"\n  Summary (n={len(seeds)}):")
        print(f"    Test Acc: {np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}%")
        print(f"    EMA Acc: {np.mean(ema_accs):.2f} ± {np.std(ema_accs):.2f}%")
        print(f"    Time: {np.mean(times):.2f} ± {np.std(times):.2f}s")

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(results_path / "ablation_summary.csv", index=False)

    # Generate visualizations
    create_visualizations(df, results_dir)

    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_path / 'ablation_summary.csv'}")
    print(f"Visualizations saved to: {results_path / 'visualizations/'}")
    print("\nSummary:")
    print(df[['configuration', 'mean_test_acc', 'std_test_acc', 'mean_training_time']].to_string(index=False))

    # Statistical analysis
    if len(seeds) >= 3:
        print(f"\n{'='*80}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*80}")

        baseline_idx = [i for i, cfg in enumerate(configurations) if cfg['name'] == 'Baseline'][0]
        baseline_acc = all_results[baseline_idx]['mean_test_acc']

        for i, result in enumerate(all_results):
            if i == baseline_idx:
                continue

            improvement = result['mean_test_acc'] - baseline_acc
            relative_improvement = 100 * improvement / baseline_acc

            print(f"\n{result['configuration']} vs Baseline:")
            print(f"  Absolute improvement: {improvement:+.2f}%")
            print(f"  Relative improvement: {relative_improvement:+.2f}%")

    return df


def create_visualizations(df: pd.DataFrame, results_dir: str):
    """
    Create comprehensive visualizations for advanced training ablation study.

    Generates:
    1. Bar plots comparing final accuracies
    2. Training curves for each configuration
    3. Feature importance heatmap
    4. Statistical significance matrix
    """
    viz_dir = Path(results_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations in {viz_dir}/...")

    # 1. Bar plot: Final test accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by configuration and compute mean/std
    grouped = df.groupby('configuration')['test_accuracy'].agg(['mean', 'std', 'count'])
    from typing import cast
    grouped = cast(pd.DataFrame, grouped).sort_values(by=['mean'], ascending=False)

    x_pos = np.arange(len(grouped))
    bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'],
                   capsize=5, alpha=0.7, edgecolor='black')

    # Color code: baseline gray, single features blue, combinations green
    colors = []
    for config in grouped.index:
        if config == 'Baseline':
            colors.append('#808080')  # Gray
        elif '+' in config:
            colors.append('#2ecc71')  # Green for combinations
        else:
            colors.append('#3498db')  # Blue for single features

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Advanced Training Features: Ablation Study Results\n(Error bars show ±1 std dev)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(grouped['mean'], grouped['std'])):
        ax.text(i, mean_val + std_val + 0.3, f'{mean_val:.2f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(viz_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(viz_dir / 'accuracy_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved accuracy_comparison.png/.pdf")

    # 2. Feature effect heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create matrix of feature combinations and their effects
    configs = grouped.index.tolist()
    baseline_acc = grouped.loc['Baseline', 'mean']

    improvements = {config: grouped.loc[config, 'mean'] - baseline_acc
                   for config in configs if config != 'Baseline'}

    # Extract feature presence
    features = ['AMP', 'Label Smoothing', 'EMA']
    feature_matrix = []
    improvement_values = []

    for config, improvement in improvements.items():
        row = [int(feat in config) for feat in features]
        feature_matrix.append(row)
        improvement_values.append(improvement)

    if feature_matrix:
        feature_matrix = np.array(feature_matrix)

        # Create heatmap
        im = ax.imshow(feature_matrix.T, cmap='RdYlGn', aspect='auto',
                      vmin=0, vmax=1, alpha=0.6)

        # Set ticks
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features, fontsize=11)
        ax.set_xticks(np.arange(len(improvements)))
        ax.set_xticklabels(list(improvements.keys()), rotation=45, ha='right', fontsize=10)

        # Add improvement values as text
        for i, (config, improvement) in enumerate(improvements.items()):
            ax.text(i, -0.5, f'+{improvement:.2f}%',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   color='green' if improvement > 0 else 'red')

        # Labels
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Active Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Activation Matrix & Performance Improvement\n(Green cells = feature active)',
                    fontsize=13, fontweight='bold')

        # Add grid
        ax.set_xticks(np.arange(len(improvements)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(features)+1)-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.savefig(viz_dir / 'feature_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(viz_dir / 'feature_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved feature_heatmap.png/.pdf")

    # 3. Box plot for variance analysis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data for box plot
    from src.utils.type_guards import ensure_series
    box_data = [ensure_series(df[df['configuration'] == config]['test_accuracy']).to_numpy()
                for config in grouped.index]

    bp = ax.boxplot(box_data, patch_artist=True, showmeans=True, meanline=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Distribution Across Seeds\n(Box = IQR, Orange line = Mean)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # Ensure x-axis labels match boxplot positions
    ax.set_xticks(np.arange(1, len(grouped.index) + 1))
    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(viz_dir / 'accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(viz_dir / 'accuracy_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved accuracy_distribution.png/.pdf")

    # 4. Training time comparison
    if 'training_time' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))

        time_grouped = df.groupby('configuration')['training_time'].agg(['mean', 'std'])
        time_grouped = time_grouped.loc[grouped.index]  # Same order as accuracy

        x_pos = np.arange(len(time_grouped))
        bars = ax.bar(x_pos, time_grouped['mean'], yerr=time_grouped['std'],
                     capsize=5, alpha=0.7, edgecolor='black', color=colors)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(time_grouped.index, rotation=45, ha='right')
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Training Time Comparison\n(Lower is better)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, mean_val in enumerate(time_grouped['mean']):
            ax.text(i, mean_val, f'{mean_val:.1f}s',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(viz_dir / 'training_time.png', dpi=300, bbox_inches='tight')
        plt.savefig(viz_dir / 'training_time.pdf', bbox_inches='tight')
        plt.close()
        print("  ✓ Saved training_time.png/.pdf")

    # 5. Summary table visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare summary table
    summary_data = []
    for config in grouped.index:
        row_data = df[df['configuration'] == config]
        summary_data.append([
            config,
            f"{grouped.loc[config, 'mean']:.2f} ± {grouped.loc[config, 'std']:.2f}",
            f"{(grouped.loc[config, 'mean'] - baseline_acc):+.2f}%",
            f"{int(grouped.loc[config, 'count'])} seeds"
        ])

    table = ax.table(cellText=summary_data,
                    colLabels=['Configuration', 'Accuracy (mean±std)', 'vs Baseline', 'Samples'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.35, 0.25, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code rows
    for i, config in enumerate(grouped.index):
        if config == 'Baseline':
            color = 'lightgray'
        elif '+' in config:
            color = 'lightgreen'
        else:
            color = 'lightblue'

        for j in range(4):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)

    # Style header
    for j in range(4):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(weight='bold', color='white')

    plt.title('Advanced Training Ablation Study: Summary Table',
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(viz_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(viz_dir / 'summary_table.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved summary_table.png/.pdf")

    print(f"\nAll visualizations saved to {viz_dir}/")

    return viz_dir


def main():
    parser = argparse.ArgumentParser(description='Advanced Training Features Ablation Study')
    parser.add_argument('--results-dir', type=str, default='results/advanced_training_ablation',
                        help='Directory to save results')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5',
                        help='Comma-separated list of random seeds')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with reduced dataset')

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    df = run_ablation_study(
        results_dir=args.results_dir,
        seeds=seeds,
        epochs=args.epochs,
        quick=args.quick
    )

    # Generate visualizations
    if df is not None and len(df) > 0:
        create_visualizations(df, args.results_dir)


if __name__ == '__main__':
    main()
