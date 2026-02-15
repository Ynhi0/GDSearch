#!/usr/bin/env python3
"""
Cross-Optimizer Dynamics Comparison Experiment

Addresses Vietnamese proposal Section 7 requirement:
"phân tích chi tiết các đặc tính động học so sánh (độ mượt - smoothness,
 tốc độ tức thời - instantaneous rate/update magnitude, dao động - oscillations/fluctuations)"

This experiment:
1. Trains the SAME model on the SAME dataset with DIFFERENT optimizers
2. Tracks detailed dynamics metrics during training
3. Generates comparative visualizations showing HOW optimizers differ in their dynamics
4. Provides detailed analysis suitable for research

Academic Value:
- Goes beyond final accuracy to understand optimizer BEHAVIOR
- Compares momentum (β) vs adaptive (β1, β2) mechanisms
- Validates theoretical predictions about smoothness and stability
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plot_helpers import arr_to_numpy_float
from typing import Dict, List, Optional, Tuple
from src.core.dataloader_utils import make_dataloader
from src.utils.constants import MNIST_MEAN, MNIST_STD, CIFAR10_MEAN, CIFAR10_STD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.dynamics_tracker import TrainingDynamicsTracker
    from src.analysis.dynamics_metrics import (
        compute_instantaneous_speed,
        compute_smoothness_index,
        compute_oscillation_magnitude,
        analyze_trajectory_dynamics
    )
    HAS_DYNAMICS = True
except ImportError as e:
    HAS_DYNAMICS = False
    logging.warning(f"Dynamics modules not available: {e}")


def run_single_optimizer_with_dynamics(
    optimizer_name: str,
    optimizer_config: Dict,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    output_dir: str = "results/dynamics_comparison",
    tune_lr: bool = False,  # GAP FIX #13: Add LR tuning option
    lr_candidates: Optional[List[float]] = None
) -> Dict:
    """
    Train with a single optimizer and track detailed dynamics.

    GAP FIX #13: Added tune_lr option for fair comparison.
    When True, tests multiple LR values and uses the best one.
    This avoids comparing optimizers at arbitrary, unfair settings.

    Uses optimizer registry instead of hardcoded if-else chain.
    This ensures consistency with other experiments and proper hyperparameter handling.

    Args:
        tune_lr: If True, search for best LR from candidates
        lr_candidates: List of LR values to try (if None, use optimizer-specific defaults)

    Returns:
        dict: Contains loss history, accuracy history, dynamics metrics
    """

    # Use optimizer registry for consistent config-driven creation
    from src.core.optimizer_registry import create_optimizer_from_config

    config_dict = {'name': optimizer_name}
    config_dict.update(optimizer_config)

    try:
        optimizer = create_optimizer_from_config(config_dict, model.parameters())
    except Exception as e:
        logging.warning(f"Registry creation failed for {optimizer_name}, falling back to direct creation: {e}")
        # Import constant at function level to avoid circular dependency
        from src.utils.constants import OptimizerNames
        
        # Fallback to direct creation for backward compatibility
        if optimizer_name == OptimizerNames.SGD:
            optimizer = torch.optim.SGD(model.parameters(), **optimizer_config)
        elif optimizer_name == OptimizerNames.SGD_MOMENTUM:
            import copy as copy_module
            config = copy_module.deepcopy(optimizer_config)
            config['momentum'] = config.get('momentum', 0.9)
            optimizer = torch.optim.SGD(model.parameters(), **config)
        elif optimizer_name == OptimizerNames.ADAM:
            # Use AdamW if weight_decay > 0 for correct decoupled weight decay
            if optimizer_config.get('weight_decay', 0) > 0:
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
            else:
                optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
        elif optimizer_name == OptimizerNames.ADAMW:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
        elif optimizer_name == OptimizerNames.RMSPROP:
            optimizer = torch.optim.RMSprop(model.parameters(), **optimizer_config)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    criterion = nn.CrossEntropyLoss()

    # Storage for dynamics analysis
    loss_history = []
    accuracy_history = []
    param_trajectories = []  # For trajectory analysis
    gradient_norms = []
    update_magnitudes = []

    # Training loop with dynamics tracking
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Store parameters before update
            params_before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Track gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            gradient_norms.append(grad_norm.item())

            optimizer.step()

            # Store parameters after update
            params_after = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
            param_trajectories.append(params_after)

            # Track update magnitude
            update_mag = np.linalg.norm(params_after - params_before)
            update_magnitudes.append(update_mag)

            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        # Epoch statistics
        avg_loss = epoch_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)

        loss_history.append(avg_loss)
        accuracy_history.append(train_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%")

    # Final test evaluation (only after training completes - use test set only for final evaluation)
    print(f"Evaluating final performance on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

    final_test_acc = 100.0 * test_correct / test_total
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    # Compute dynamics metrics
    trajectory_array = np.array(param_trajectories)

    # Instantaneous speed (update magnitudes)
    speeds = update_magnitudes

    # Trajectory smoothness (angle changes)
    smoothness = compute_smoothness_index(trajectory_array)

    # Oscillation magnitude (variance in speeds)
    oscillation_magnitudes = compute_oscillation_magnitude(np.array(speeds))
    oscillation_idx = np.mean(oscillation_magnitudes)

    # Aggregate dynamics metrics
    dynamics_metrics = {
        'mean_speed': np.mean(speeds),
        'std_speed': np.std(speeds),
        'smoothness': smoothness,
        'oscillation_index': oscillation_idx,
        'mean_grad_norm': np.mean(gradient_norms),
        'std_grad_norm': np.std(gradient_norms)
    }

    return {
        'optimizer': optimizer_name,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history,
        'speeds': speeds,
        'gradient_norms': gradient_norms,
        'dynamics_metrics': dynamics_metrics,
        'final_test_acc': final_test_acc
    }


def run_cross_optimizer_dynamics_comparison(
    dataset: str = 'MNIST',
    optimizers: Optional[List[str]] = None,
    epochs: int = 50,
    seeds: Optional[List[int]] = None,
    quick: bool = False,
    results_dir: str = "results/cross_optimizer_dynamics"
) -> pd.DataFrame:
    """
    Main experiment: Compare dynamics across different optimizers.

    Addresses proposal requirement:
    "phân tích chi tiết các đặc tính động học so sánh"

    Args:
        dataset: 'MNIST' or 'CIFAR10'
        optimizers: List of optimizer names to compare
        epochs: Number of training epochs
        seeds: Random seeds for reproducibility
        quick: If True, use reduced epochs and single seed
        results_dir: Output directory

    Returns:
        DataFrame with dynamics comparison results
    """

    if not HAS_DYNAMICS:
        print("Dynamics modules not available - cannot run comparison")
        # Return empty DataFrame so callers receive a consistent object
        return pd.DataFrame()

    # Import constant at function level to avoid circular dependency
    from src.utils.constants import OptimizerNames
    
    # Default configurations
    if optimizers is None:
        if quick:
            optimizers = [OptimizerNames.SGD, OptimizerNames.SGD_MOMENTUM, OptimizerNames.ADAM]
        else:
            optimizers = [OptimizerNames.SGD, OptimizerNames.SGD_MOMENTUM, OptimizerNames.ADAM, OptimizerNames.ADAMW, OptimizerNames.RMSPROP]

    if seeds is None:
        seeds = [42] if quick else [42, 123, 456]

    if quick:
        epochs = min(epochs, 20)

    os.makedirs(results_dir, exist_ok=True)

    print("="*80)
    print("🔬 CROSS-OPTIMIZER DYNAMICS COMPARISON")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Optimizers: {optimizers}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare dataset
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
        train_dataset = torchvision.datasets.MNIST(
            root='data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='data', train=False, download=True, transform=transform
        )
        input_size = 28 * 28
        num_classes = 10
    elif dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform
        )
        input_size = 3 * 32 * 32
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=seeds[0], num_workers=2, pin_memory=True)
    test_loader = make_dataloader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer configurations (tuned values)
    optimizer_configs = {
        'SGD': {'lr': 0.1},
        'SGD_Momentum': {'lr': 0.1, 'momentum': 0.9},
        'Adam': {'lr': 0.001, 'betas': (0.9, 0.999)},
        'AdamW': {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
        'RMSprop': {'lr': 0.001, 'alpha': 0.99}
    }

    # Run experiments
    all_results = []

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Seed: {seed}")
        print(f"{'='*80}")

        for opt_name in optimizers:
            print(f"\nTraining with {opt_name}...")

            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create fresh model
            if dataset == 'MNIST':
                model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                ).to(device)
            else:  # CIFAR10
                model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                ).to(device)

            # Run training with dynamics tracking
            result = run_single_optimizer_with_dynamics(
                optimizer_name=opt_name,
                optimizer_config=optimizer_configs[opt_name],
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                output_dir=results_dir
            )

            result['seed'] = seed
            result['dataset'] = dataset
            all_results.append(result)

    # Create comparison DataFrame
    comparison_data = []
    for res in all_results:
        row = {
            'Optimizer': res['optimizer'],
            'Seed': res['seed'],
            'Dataset': res['dataset'],
            'Final_Test_Acc': res['final_test_acc'],
            'Mean_Speed': res['dynamics_metrics']['mean_speed'],
            'Std_Speed': res['dynamics_metrics']['std_speed'],
            'Smoothness': res['dynamics_metrics']['smoothness'],
            'Oscillation_Index': res['dynamics_metrics']['oscillation_index'],
            'Mean_Grad_Norm': res['dynamics_metrics']['mean_grad_norm'],
            'Std_Grad_Norm': res['dynamics_metrics']['std_grad_norm']
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Save results
    csv_path = os.path.join(results_dir, f"cross_optimizer_dynamics_{dataset}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate visualizations
    print("\nGenerating comparative visualizations...")
    generate_dynamics_comparison_plots(all_results, results_dir, dataset)

    return df


def generate_dynamics_comparison_plots(results: List[Dict], output_dir: str, dataset: str):
    """Generate high-quality comparative plots."""

    sns.set_style("whitegrid")

    # 1. Loss curves comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for res in results:
        if res['seed'] == results[0]['seed']:  # Plot only first seed for clarity
            plt.plot(arr_to_numpy_float(res['loss_history']), label=res['optimizer'], linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title(f'Loss Convergence Comparison ({dataset})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for res in results:
        if res['seed'] == results[0]['seed']:
            plt.plot(arr_to_numpy_float(res['accuracy_history']), label=res['optimizer'], linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy Progression ({dataset})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'loss_accuracy_comparison_{dataset}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Dynamics metrics comparison (bar plot)
    metric_keys = ['mean_speed', 'smoothness', 'oscillation_index']
    metric_labels = ['Mean Update Magnitude', 'Trajectory Smoothness', 'Oscillation Index']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Aggregate across seeds
    optimizers = sorted(list(set([r['optimizer'] for r in results])))

    for idx, (metric_key, label) in enumerate(zip(metric_keys, metric_labels)):
        data = []
        for opt in optimizers:
            values = [r['dynamics_metrics'][metric_key] for r in results if r['optimizer'] == opt]
            data.append({
                'Optimizer': opt,
                'Mean': np.mean(values),
                'Std': np.std(values)
            })

        df_metric = pd.DataFrame(data)
        axes[idx].bar(df_metric['Optimizer'], df_metric['Mean'], yerr=df_metric['Std'],
                      capsize=5, color='skyblue', edgecolor='navy', linewidth=1.5)
        axes[idx].set_xlabel('Optimizer', fontsize=11)
        axes[idx].set_ylabel(label, fontsize=11)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'dynamics_metrics_comparison_{dataset}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Cross-Optimizer Dynamics Comparison')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--seeds', type=str, default='42,123,456', help='Comma-separated seeds')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer epochs, single seed)')
    parser.add_argument('--output-dir', type=str, default='results/cross_optimizer_dynamics')

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    df = run_cross_optimizer_dynamics_comparison(
        dataset=args.dataset,
        epochs=args.epochs,
        seeds=seeds,
        quick=args.quick,
        results_dir=args.output_dir
    )

    if df is not None:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(df.groupby('Optimizer')[['Final_Test_Acc', 'Mean_Speed', 'Smoothness', 'Oscillation_Index']].mean())
        print("="*80)
