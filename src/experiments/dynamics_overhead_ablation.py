"""
Dynamics Tracking Overhead Ablation Study

This module quantifies the computational cost of using DynamicsTracker
during neural network training. Required for academic rigor - cannot claim
"negligible overhead" without evidence.

Academic Value:
    Measures the trade-off between detailed trajectory analysis and training speed.
    Validates that monitoring does not affect final convergence quality.

Author: GDSearch Team
Date: December 7, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.core.dataloader_utils import make_dataloader
import psutil
import matplotlib.pyplot as plt


# Import dynamics tracker
try:
    from src.core.dynamics_tracker import TrainingDynamicsTracker
    HAS_DYNAMICS_TRACKER = True
except ImportError:
    HAS_DYNAMICS_TRACKER = False
    print("TrainingDynamicsTracker not available - ablation study limited")


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST - same as used in main experiments"""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def train_with_optional_tracking(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    use_dynamics_tracker: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Train model with or without dynamics tracking.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: PyTorch optimizer
        device: Device to train on
        epochs: Number of epochs
        use_dynamics_tracker: Whether to use DynamicsTracker
        output_dir: Directory for saving tracker outputs
        
    Returns:
        Dictionary with metrics: time, memory, final accuracy
    """
    criterion = nn.CrossEntropyLoss()
    
    # Initialize tracker if requested
    tracker = None
    if use_dynamics_tracker and HAS_DYNAMICS_TRACKER:
        tracker = TrainingDynamicsTracker()  # output paths handled at save time
    
    # Memory tracking
    process = psutil.Process()
    mem_before_mb = process.memory_info().rss / (1024 ** 2)
    gpu_mem_before_mb = get_gpu_memory_usage()
    
    # Time tracking
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Track dynamics if enabled
            if tracker is not None:
                    tracker.track_step(
                        iteration=epoch * len(train_loader) + batch_idx,
                        loss=loss.item(),
                        model=model,
                        optimizer=optimizer,
                    )
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    final_accuracy = 100.0 * correct / max(1, total)
    
    # Compute metrics
    elapsed_time = time.time() - start_time
    mem_after_mb = process.memory_info().rss / (1024 ** 2)
    gpu_mem_after_mb = get_gpu_memory_usage()
    
    # Save tracker visualizations if enabled
    if tracker is not None:
        try:
            out_dir = Path(output_dir or "dynamics_tmp")
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save CSV of dynamics and plots
            tracker.save_dynamics(str(out_dir / "dynamics.csv"))
            tracker.plot_dynamics(str(out_dir / "plots"), optimizer_name=optimizer.__class__.__name__)
        except Exception as e:
            print(f"   Tracker save failed: {e}")
    
    return {
        'time_seconds': elapsed_time,
        'memory_mb': mem_after_mb - mem_before_mb,
        'gpu_memory_mb': gpu_mem_after_mb - gpu_mem_before_mb,
        'final_accuracy': final_accuracy
    }


def run_dynamics_overhead_ablation(
    dataset: str = 'MNIST',
    model_name: str = 'SimpleMLP',
    optimizer_name: str = 'Adam',
    epochs: int = 5,
    seeds: List[int] = [42, 123, 456, 789, 1011],
    results_dir: str = 'results/dynamics_overhead_ablation',
    quick: bool = False
) -> pd.DataFrame:
    """
    Run ablation study comparing training WITH vs WITHOUT dynamics tracking.
    
    This study addresses the question: "Does DynamicsTracker add significant
    computational overhead?" Required for academic rigor.
    
    Args:
        dataset: Dataset name (currently only MNIST supported)
        model_name: Model architecture
        optimizer_name: Optimizer to use
        epochs: Training epochs
        seeds: Random seeds for statistical significance
        results_dir: Output directory
        quick: Quick mode (fewer seeds)
        
    Returns:
        DataFrame with ablation results
    """
    print("\n" + "="*80)
    print("🔬 DYNAMICS TRACKING OVERHEAD ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Epochs: {epochs}")
    print(f"Seeds: {len(seeds)}")
    print()
    
    if not HAS_DYNAMICS_TRACKER:
        print("TrainingDynamicsTracker not available - cannot run ablation")
        return pd.DataFrame()
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    
    # Use subset for quick mode
    if quick:
        train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
        test_dataset = torch.utils.data.Subset(test_dataset, range(2000))
        seeds = seeds[:3]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    # Run for each seed
    for seed_idx, seed in enumerate(seeds):
        print(f"\n[{seed_idx+1}/{len(seeds)}] Running seed {seed}...")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create dataloaders
        train_loader = make_dataloader(
            train_dataset, batch_size=128, shuffle=True, seed=seed,
            num_workers=2, pin_memory=True if device.type == 'cuda' else False
        )
        test_loader = make_dataloader(
            test_dataset, batch_size=256, shuffle=False,
            num_workers=2, pin_memory=True if device.type == 'cuda' else False
        )
        
        # Test 1: WITHOUT dynamics tracking (baseline)
        print("   Testing WITHOUT dynamics tracking (baseline)...")
        model_baseline = SimpleMLP().to(device)
        optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=1e-3)
        
        metrics_baseline = train_with_optional_tracking(
            model=model_baseline,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer_baseline,
            device=device,
            epochs=epochs,
            use_dynamics_tracker=False
        )
        
        # Test 2: WITH dynamics tracking
        print("   Testing WITH dynamics tracking...")
        model_tracked = SimpleMLP().to(device)
        optimizer_tracked = optim.Adam(model_tracked.parameters(), lr=1e-3)
        
        tracker_dir = os.path.join(results_dir, f"dynamics_seed{seed}")
        
        metrics_tracked = train_with_optional_tracking(
            model=model_tracked,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer_tracked,
            device=device,
            epochs=epochs,
            use_dynamics_tracker=True,
            output_dir=tracker_dir
        )
        
        # Compute overhead
        time_overhead_pct = (
            (metrics_tracked['time_seconds'] - metrics_baseline['time_seconds']) /
            metrics_baseline['time_seconds'] * 100
        )
        mem_overhead_mb = metrics_tracked['memory_mb'] - metrics_baseline['memory_mb']
        acc_diff = metrics_tracked['final_accuracy'] - metrics_baseline['final_accuracy']
        
        # Store results
        results.append({
            'seed': seed,
            'condition': 'baseline',
            'time_seconds': metrics_baseline['time_seconds'],
            'memory_mb': metrics_baseline['memory_mb'],
            'gpu_memory_mb': metrics_baseline['gpu_memory_mb'],
            'accuracy': metrics_baseline['final_accuracy']
        })
        
        results.append({
            'seed': seed,
            'condition': 'with_tracking',
            'time_seconds': metrics_tracked['time_seconds'],
            'memory_mb': metrics_tracked['memory_mb'],
            'gpu_memory_mb': metrics_tracked['gpu_memory_mb'],
            'accuracy': metrics_tracked['final_accuracy']
        })
        
        print(f"   ⏱️  Time overhead: {time_overhead_pct:+.2f}%")
        print(f"   💾 Memory overhead: {mem_overhead_mb:+.2f} MB")
        print(f"   🎯 Accuracy difference: {acc_diff:+.4f}%")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = os.path.join(results_dir, f"dynamics_overhead_ablation_{dataset}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    baseline_df = df[df['condition'] == 'baseline']
    tracked_df = df[df['condition'] == 'with_tracking']
    
    time_overhead_mean = (
        (tracked_df['time_seconds'].mean() - baseline_df['time_seconds'].mean()) /
        baseline_df['time_seconds'].mean() * 100
    )
    mem_overhead_mean = tracked_df['memory_mb'].mean() - baseline_df['memory_mb'].mean()
    acc_diff_mean = tracked_df['accuracy'].mean() - baseline_df['accuracy'].mean()
    
    print(f"Average Time Overhead: {time_overhead_mean:+.2f}%")
    print(f"Average Memory Overhead: {mem_overhead_mean:+.2f} MB")
    print(f"Average Accuracy Difference: {acc_diff_mean:+.4f}%")
    
    # Statistical test with proper seed alignment
    from scipy import stats
    # Ensure DataFrame types and Merge on seed to ensure proper pairing for paired t-test
    from src.utils.type_guards import ensure_dataframe
    baseline_df = ensure_dataframe(baseline_df)
    tracked_df = ensure_dataframe(tracked_df)
    merged = pd.merge(
        baseline_df[['seed', 'time_seconds', 'accuracy']],
        tracked_df[['seed', 'time_seconds', 'accuracy']],
        on='seed',
        how='inner',
        suffixes=('_baseline', '_tracked')
    )
    
    if len(merged) < 2:
        print(f"\nWarning: Insufficient paired samples (n={len(merged)}). Skipping statistical tests.")
        time_ttest = None
        acc_ttest = None
    else:
        time_ttest = stats.ttest_rel(merged['time_seconds_tracked'], merged['time_seconds_baseline'])
        acc_ttest = stats.ttest_rel(merged['accuracy_tracked'], merged['accuracy_baseline'])
    
    print(f"\nStatistical Significance:")
    if time_ttest is not None and acc_ttest is not None:
        print(f"  Time difference: p={time_ttest.pvalue:.4f} {'(significant)' if time_ttest.pvalue < 0.05 else '(not significant)'}")
        print(f"  Accuracy difference: p={acc_ttest.pvalue:.4f} {'(significant)' if acc_ttest.pvalue < 0.05 else '(not significant)'}")
    else:
        print(f"  Statistical tests skipped due to insufficient paired samples.")
    
    # Generate visualization
    try:
        create_ablation_visualization(df, results_dir)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    return df


def create_ablation_visualization(df: pd.DataFrame, output_dir: str):
    """Create high-quality visualization of ablation results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    
    baseline_df = df[df['condition'] == 'baseline']
    tracked_df = df[df['condition'] == 'with_tracking']
    
    # Plot 1: Training time comparison
    axes[0].bar(['Baseline', 'With Tracking'], 
                [baseline_df['time_seconds'].mean(), tracked_df['time_seconds'].mean()],
                yerr=[baseline_df['time_seconds'].std(), tracked_df['time_seconds'].std()],
                capsize=5, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('Training Time (seconds)')
    axes[0].set_title('Training Time Overhead')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Memory overhead
    axes[1].bar(['Baseline', 'With Tracking'],
                [baseline_df['memory_mb'].mean(), tracked_df['memory_mb'].mean()],
                yerr=[baseline_df['memory_mb'].std(), tracked_df['memory_mb'].std()],
                capsize=5, color=['#3498db', '#e74c3c'])
    axes[1].set_ylabel('Memory Usage (MB)')
    axes[1].set_title('Memory Overhead')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Final accuracy (should be identical)
    axes[2].bar(['Baseline', 'With Tracking'],
                [baseline_df['accuracy'].mean(), tracked_df['accuracy'].mean()],
                yerr=[baseline_df['accuracy'].std(), tracked_df['accuracy'].std()],
                capsize=5, color=['#3498db', '#e74c3c'])
    axes[2].set_ylabel('Test Accuracy (%)')
    axes[2].set_title('Final Accuracy (Should be Equal)')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dynamics_overhead_ablation.png'), dpi=300, bbox_inches='tight')
    print(f"   Visualization saved to {output_dir}/dynamics_overhead_ablation.png")


if __name__ == '__main__':
    # Run ablation study
    df = run_dynamics_overhead_ablation(
        epochs=5,
        seeds=[42, 123, 456, 789, 1011],
        quick=False
    )
    print("\nAblation study complete!")
