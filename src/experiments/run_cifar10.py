#!/usr/bin/env python3
"""
CIFAR-10 multi-seed benchmark with ResNet-18.

ARCHITECTURE STANDARDIZATION (Dec 2025):
    - Previously used SimpleCIFARNet (toy model: 2 conv layers, ~0.5M params)
    - NOW uses ResNet-18 (industry standard: 18 layers, ~11M params)
    - Reason: Match Kaggle benchmarks for valid cross-comparison
    - SimpleCIFARNet deprecated but available in models.py as SimpleCNN

GAP 43 NOTE - OPTIMIZER IMPLEMENTATION:
    This benchmark uses PyTorch's built-in optimizer implementations (torch.optim)
    for performance reasons. This is the "PyTorch Baseline" benchmark.

    For experiments using custom GDSearch implementations:
        from src.core.pytorch_optimizers import SGDWrapper, AdamWrapper
        optimizer = AdamWrapper(model.parameters(), lr=0.001)

    The custom implementations in src.core.optimizers.py are educational/prototype
    versions that are tested in tests/test_optimizers.py.

Outputs per-run CSVs compatible with the project's result conventions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import logging
from src.utils.constants import CIFAR10_MEAN, CIFAR10_STD

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from src.core.training_utils import set_seed
from src.core.models import ResNet18  # ARCHITECTURE STANDARDIZATION

# NOTE: SimpleCIFARNet removed - use ResNet18 for all CIFAR-10 benchmarks
# For legacy compatibility, SimpleCNN is available in src.core.models


def get_loaders(batch_size: int = 128, seed: int = 42):
    """
    Get CIFAR-10 data loaders.

    GAP 48 FIX: Now accepts seed parameter for proper randomization.
    Each experiment seed should produce different batch orderings.

    Args:
        batch_size: Batch size for training
        seed: Random seed for dataloader shuffling (should match experiment seed)
    """
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    root = os.environ.get('DATA_ROOT', './data')
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    # Use make_dataloader for consistent settings
    # GAP 48 FIX: Pass experiment seed to dataloader for proper variance analysis
    # Previously hardcoded seed=42 meant all experiments had identical batch ordering
    from src.core.dataloader_utils import make_dataloader
    trainloader = make_dataloader(trainset, batch_size=batch_size, shuffle=True, seed=seed, num_workers=4, pin_memory=True)
    testloader = make_dataloader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader


def train_one_epoch(model, loader, optimizer, device):
    """
    Train model for one epoch.

    GAP 51 FIX: Now returns gradient norm for convergence analysis.
    GAP 52 FIX: Returns both running_avg_loss (historical) and train_eval_loss (current state).

    Returns:
        Tuple of (avg_train_loss, train_accuracy, gradient_norm)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # GAP 51 FIX: Track gradient norm for convergence analysis
    grad_norm_sum = 0.0
    grad_norm_count = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()

        # GAP 51 FIX: Compute gradient norm BEFORE optimizer.step()
        # This is critical for convergence analysis (||∇f|| → 0 at stationary points)
        batch_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                batch_grad_norm += p.grad.data.norm(2).item() ** 2
        batch_grad_norm = batch_grad_norm ** 0.5
        grad_norm_sum += batch_grad_norm
        grad_norm_count += 1

        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    # Avoid division by zero
    if total == 0:
        return 0.0, 0.0, 0.0

    avg_grad_norm = grad_norm_sum / grad_norm_count if grad_norm_count > 0 else 0.0
    return total_loss / total, correct / total, avg_grad_norm


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    # Avoid division by zero
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def run_single(optimizer_name: str, seed: int, lr: float, epochs: int, batch_size: int, results_dir: Path):
    """
    Run a single CIFAR-10 training experiment.

    GAP 47 FIX: Standardized weight_decay across optimizers for fair comparison.
    GAP 48 FIX: Pass seed to dataloader for proper batch order randomization.
    GAP 51 FIX: Log gradient norms for convergence analysis.
    GAP 52 FIX: Log train_eval_loss (current state, not running average).
    GAP 53 FIX: Log current learning rate.
    GAP 54 FIX: Added SGD_Nesterov option.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # GAP 48 FIX: Pass seed to get_loaders for proper variance analysis
    trainloader, testloader = get_loaders(batch_size, seed=seed)
    model = ResNet18().to(device)  # ARCHITECTURE STANDARDIZATION: Use ResNet18 instead of SimpleCIFARNet

    # Import constant at function level to avoid circular dependency
    from src.utils.constants import OptimizerNames
    
    # GAP 47 FIX: Standardize weight_decay for fair optimizer comparison
    # All optimizers get same regularization to compare algorithms, not regularization strength
    weight_decay = 5e-4  # Standard for CIFAR-10 ResNet

    if optimizer_name == OptimizerNames.SGD:
        # GAP 47 FIX: Add weight_decay for fair comparison with AdamW
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == OptimizerNames.SGD_MOMENTUM:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == OptimizerNames.SGD_NESTEROV:
        # GAP 54 FIX: Added Nesterov Accelerated Gradient
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif optimizer_name == OptimizerNames.ADAM:
        # Use AdamW for decoupled weight decay when weight_decay > 0 (Loshchilov & Hutter 2019)
        # Original Adam couples weight decay with adaptive LR, causing effective regularization
        # to vary by ~100x across parameters (incorrect behavior)
        if weight_decay > 0:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif optimizer_name == OptimizerNames.ADAMW:
        # AdamW uses decoupled weight decay (different from L2 in Adam)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == OptimizerNames.RMSPROP:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    hist = []
    for epoch in range(1, epochs + 1):
        # GAP 51 FIX: train_one_epoch now returns gradient norm
        tr_loss, tr_acc, grad_norm = train_one_epoch(model, trainloader, optimizer, device)

        # GAP 52 FIX: Compute train_eval_loss (current model state, not running average)
        # This is mathematically correct for optimization analysis: f(θ_k)
        train_eval_loss, _ = evaluate(model, trainloader, device)  # train_eval_acc unused

        te_loss, te_acc = evaluate(model, testloader, device)

        # GAP 53 FIX: Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        hist.append({
            'epoch': epoch,
            'train_loss': tr_loss,  # Historical (running average during epoch)
            'train_eval_loss': train_eval_loss,  # GAP 52 FIX: Current state f(θ_k)
            'train_acc': tr_acc,
            'test_loss': te_loss,
            'test_accuracy': te_acc,  # AUDIT FIX: Renamed from test_acc for aggregator compatibility
            'grad_norm': grad_norm,  # GAP 51 FIX: For convergence analysis
            'learning_rate': current_lr  # GAP 53 FIX: Track LR changes
        })
        print(f"seed={seed} {optimizer_name} [{epoch}/{epochs}] train_acc={tr_acc:.3f} test_acc={te_acc:.3f} grad_norm={grad_norm:.4f}")

    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    df = pd.DataFrame(hist)
    df['elapsed_seconds'] = elapsed
    df['peak_gpu_mb'] = peak_mb

    # AUDIT FIX: Ensure results directory exists before writing
    results_dir.mkdir(parents=True, exist_ok=True)

    # Updated naming: ResNet18 instead of SimpleCIFAR10
    out = results_dir / f"NN_ResNet18_CIFAR10_{optimizer_name}_lr{lr}_seed{seed}.csv"
    df.to_csv(out, index=False)
    return out


def _get_epochs_and_test_acc(df):
    """Return (epochs, test_accuracy_series_in_percent_or_None).

    Robust extraction of epoch and test accuracy from different CSV column naming conventions.
    """
    epoch_col = 'epoch' if 'epoch' in df.columns else next((col for col in df.columns if 'epoch' in col.lower()), None)
    if epoch_col:
        epochs = df[epoch_col].values
    else:
        epochs = np.arange(1, len(df) + 1)

    acc_col = next((col for col in df.columns if 'test' in col.lower() and ('acc' in col.lower() or 'accuracy' in col.lower())), None)
    if acc_col is None:
        return epochs, None

    acc_vals = pd.to_numeric(df[acc_col], errors='coerce')
    if acc_vals.isna().all():
        return epochs, None

    if acc_vals.max() <= 1.01:
        acc_vals = acc_vals * 100.0

    return epochs, acc_vals.values


def create_cifar10_summary_plots(results_dir: Path, output_file: str = 'cifar10_summary.png'):
    """
    Create high-quality summary plots from CIFAR-10 results.
    Generates learning curves comparing all optimizers across seeds.

    GAP 55 FIX: Uses log-scale for loss plots to reveal convergence rate differences.
    On linear scale, O(1/k) and O(1/k²) look identical after first few epochs.
    """
    import matplotlib.pyplot as plt  # Import locally for function use
    import glob

    # Find all result CSVs
    # Support both legacy (SimpleCIFAR10) and new (ResNet18) naming
    csv_files = glob.glob(str(results_dir / "NN_ResNet18_CIFAR10_*.csv"))
    if not csv_files:
        # Fallback to legacy naming for old results
        csv_files = glob.glob(str(results_dir / "NN_SimpleCIFAR10_*.csv"))

    if not csv_files:
        print("No CIFAR-10 results found for visualization")
        return

    # Parse results
    results = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract optimizer and seed from filename
        basename = os.path.basename(csv_file)
        # Format: NN_ResNet18_CIFAR10_{optimizer}_lr{lr}_seed{seed}.csv
        # Legacy: NN_SimpleCIFAR10_{optimizer}_lr{lr}_seed{seed}.csv
        parts = basename.replace('.csv', '').split('_')

        # Find optimizer name (after model architecture identifier, before lr)
        if 'ResNet18' in parts and 'CIFAR10' in parts:
            opt_start = parts.index('CIFAR10') + 1  # New format
        elif 'SimpleCIFAR10' in parts:
            opt_start = parts.index('SimpleCIFAR10') + 1  # Legacy format
        else:
            print(f"Skipping unrecognized format: {basename}")
            continue

        opt_parts = []
        for i in range(opt_start, len(parts)):
            if parts[i].startswith('lr'):
                break
            opt_parts.append(parts[i])
        optimizer = '_'.join(opt_parts)

        # Find seed
        seed_part = [p for p in parts if p.startswith('seed')]
        seed = int(seed_part[0].replace('seed', '')) if seed_part else 0

        if optimizer not in results:
            results[optimizer] = []
        results[optimizer].append((seed, df))

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CIFAR-10 Training Results', fontsize=16, fontweight='bold')

    colors = {'SGD': '#FF6B6B', 'SGD_Momentum': '#4ECDC4', 'Adam': '#45B7D1',
              'AdamW': '#FFA07A', 'RMSProp': '#98D8C8'}

    # Plot 1: Train Loss (GAP 55 FIX: Log scale for convergence rate visibility)
    ax = axes[0, 0]
    for optimizer, runs in results.items():
        for seed, df in runs:
            color = colors.get(optimizer, '#999999')
            alpha = 0.3 if len(runs) > 1 else 1.0
            ax.plot(df['epoch'], df['train_loss'], color=color, alpha=alpha, linewidth=1)
        # Plot mean
        if len(runs) > 1:
            epochs = runs[0][1]['epoch'].values
            losses = np.array([df['train_loss'].values for _, df in runs])
            mean_loss = losses.mean(axis=0)
            ax.plot(epochs, mean_loss, color=colors.get(optimizer, '#999999'),
                   linewidth=2.5, label=optimizer)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Train Loss (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss (Log Scale)', fontsize=13, fontweight='bold')
    # GAP 55 FIX: Use log scale to reveal convergence rate differences
    # On linear scale, O(1/k) and O(1/k²) rates look nearly identical
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for optimizer, runs in results.items():
        plot_runs = []
        for seed, df in runs:
            epochs, acc = _get_epochs_and_test_acc(df)
            if acc is not None:
                color = colors.get(optimizer, '#999999')
                alpha = 0.3 if len(runs) > 1 else 1.0
                ax.plot(epochs, acc, color=color, alpha=alpha, linewidth=1)
                plot_runs.append((epochs, acc))
        # Plot mean
        if len(plot_runs) > 0:
            common_epochs = np.arange(1, int(max(e.max() for e, _ in plot_runs)) + 1)
            aligned = []
            for e, a in plot_runs:
                s = pd.Series(a, index=e)
                s = s.reindex(common_epochs).interpolate().ffill().bfill().values
                aligned.append(s)
            mean_acc = np.mean(np.vstack(aligned), axis=0)
            ax.plot(common_epochs, mean_acc, color=colors.get(optimizer, '#999999'),
                   linewidth=2.5, label=optimizer)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Final Test Accuracy (Bar plot)
    ax = axes[1, 0]
    final_accs = {}
    final_stds = {}
    for optimizer, runs in results.items():
        acc_vals = []
        for _, df in runs:
            _, acc = _get_epochs_and_test_acc(df)
            if acc is not None:
                acc_vals.append(acc[-1])
        if acc_vals:
            final_accs[optimizer] = np.mean(acc_vals)
            final_stds[optimizer] = np.std(acc_vals) if len(acc_vals) > 1 else 0

    optimizers_sorted = sorted(final_accs.keys(), key=lambda x: final_accs[x], reverse=True)
    x_pos = np.arange(len(optimizers_sorted))
    bars = ax.bar(x_pos, [final_accs[opt] for opt in optimizers_sorted],
                  yerr=[final_stds[opt] for opt in optimizers_sorted],
                  color=[colors.get(opt, '#999999') for opt in optimizers_sorted],
                  alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(optimizers_sorted, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (opt, bar) in enumerate(zip(optimizers_sorted, bars)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{final_accs[opt]:.2f}%\n±{final_stds[opt]:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 4: Training Speed (samples/sec)
    ax = axes[1, 1]
    speeds = {}
    for optimizer, runs in results.items():
        # Estimate samples/sec from elapsed time and epochs
        run_speeds = []
        for seed, df in runs:
            if 'elapsed_seconds' in df.columns and df['elapsed_seconds'].iloc[0] > 0:
                total_epochs = df['epoch'].max()
                elapsed = df['elapsed_seconds'].iloc[0]
                # Assuming 50k training samples
                samples_per_sec = (50000 * total_epochs) / elapsed
                run_speeds.append(samples_per_sec)
        if run_speeds:
            speeds[optimizer] = np.mean(run_speeds)

    if speeds:
        optimizers_sorted = sorted(speeds.keys(), key=lambda x: speeds[x], reverse=True)
        x_pos = np.arange(len(optimizers_sorted))
        ax.bar(x_pos, [speeds[opt] for opt in optimizers_sorted],
               color=[colors.get(opt, '#999999') for opt in optimizers_sorted],
               alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizers_sorted, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Training Speed (samples/sec)', fontsize=12, fontweight='bold')
        ax.set_title('Training Efficiency', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No timing data available',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)

    plt.tight_layout()
    output_path = results_dir / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"CIFAR-10 summary plot saved: {output_path}")
    plt.close()

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CIFAR-10 Multi-Seed Runner')
    parser.add_argument('--seeds', type=str, default='1,2,3')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--quick', action='store_true')
    args, _ = parser.parse_known_args()

    seeds = [1, 2, 3] if args.quick else [int(s) for s in args.seeds.split(',') if s]
    epochs = 2 if args.quick else args.epochs
    batch_size = args.batch_size

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

        # GAP 54 FIX: Added SGD_Nesterov to optimizer config
    # GAP 47 FIX: Learning rates tuned for fair comparison with standardized weight_decay
    opt_config = [
        ('SGD', 0.1),
        ('SGD_Momentum', 0.1),
        ('SGD_Nesterov', 0.1),  # GAP 54 FIX: Nesterov Accelerated Gradient
        ('Adam', 1e-3),
        ('AdamW', 1e-3),
        ('RMSProp', 1e-3)
    ]

    completed = 0
    for opt, lr in opt_config:
        for seed in seeds:
            try:
                run_single(opt, seed, lr, epochs, batch_size, results_dir)
                completed += 1
            except (RuntimeError, ValueError, OSError) as e:
                print('Error:', e)

    print(f"Completed {completed} runs")

    # Generate summary visualization
    if completed > 0:
        try:
            create_cifar10_summary_plots(results_dir)
        except (FileNotFoundError, ValueError, ImportError) as e:
            print(f"Failed to create summary plots: {e}")


if __name__ == '__main__':
    main()
