#!/usr/bin/env python3
"""
Optimizer-Initialization Interaction Ablation Study

Academic Question:
How do different weight initialization strategies interact with various optimizers?

Research Motivation:
- Different optimizers may be more/less sensitive to initialization
- Modern initializations (Kaiming/He, Xavier/Glorot) were designed for specific activations
- Understanding these interactions helps practitioners make better choices

Experimental Design:
1. Test multiple initialization schemes:
   - Zero initialization (pathological baseline)
   - Uniform random
   - Normal random (std=0.01)
   - Xavier/Glorot (uniform and normal)
   - Kaiming/He (uniform and normal)

2. Test with multiple optimizers:
   - SGD (sensitive to initialization)
   - SGD+Momentum  
   - Adam (more robust to initialization)
   - AdamW

3. Multiple seeds for statistical validity

4. Measure:
   - Convergence speed (epochs to reach threshold)
   - Final accuracy
   - Training stability (variance across seeds)

Expected Findings:
- Adaptive optimizers (Adam/AdamW) should be more robust to poor initialization
- SGD should be more sensitive to initialization quality
- Kaiming init should work best for ReLU networks
- Xavier init should work best for Tanh/Sigmoid networks
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
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import make_dataloader
from src.core.dataloader_utils import make_dataloader

try:
    from src.visualization.ablation_plots import generate_all_ablation_plots
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.visualization.ablation_plots import generate_all_ablation_plots

from src.core.training_utils import set_seed
from src.core.models import SimpleCNN  # Import from central models.py


# Removed duplicate set_seed - using from src.core.training_utils
# Removed duplicate SimpleCNN - using from src.core.models


# SCIENTIFIC FIX: Centralized model definitions
# Previously, this file defined its own SimpleCNN class, creating two problems:
# 1. CONSISTENCY: If we fix initialization bugs in src.core.models.py,
#    this ablation script wouldn't see the fix (running on a different model).
# 2. VALIDITY: "Ablation" means removing ONE variable from the MAIN experiment.
#    Using a different model class invalidates the ablation claim.
#
# SOLUTION: Import SimpleCNN from src.core.models.py to ensure all experiments
# use the same model architecture.


def apply_custom_initialization(model: nn.Module, init_method: str):
    """Apply specified initialization to model layers.
    
    This function replaces the previous SimpleCNN.apply_initialization() method.
    It works with any model by targeting Conv2d, Linear layers.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_method == 'zero':
                # Pathological case - all zeros
                if hasattr(module, 'weight'):
                    nn.init.constant_(module.weight, 0.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            
            elif init_method == 'uniform_small':
                # Uniform in [-0.1, 0.1]
                if hasattr(module, 'weight'):
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)
            
            elif init_method == 'normal_small':
                # Normal with std=0.01
                if hasattr(module, 'weight'):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.01)
            
            elif init_method == 'xavier_uniform':
                # Xavier/Glorot uniform
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            
            elif init_method == 'xavier_normal':
                # Xavier/Glorot normal
                if hasattr(module, 'weight'):
                    nn.init.xavier_normal_(module.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            
            elif init_method == 'kaiming_uniform':
                # Kaiming/He uniform
                if hasattr(layer, 'weight'):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            
            elif init_method == 'kaiming_normal':
                # Kaiming/He normal
                if hasattr(layer, 'weight'):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            
            else:
                raise ValueError(f"Unknown initialization method: {init_method}")


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def run_single_experiment(
    init_method: str,
    optimizer_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 0.001,
    seed: int = 42,
    activation: str = 'relu'
) -> Dict:
    """Run a single initialization-optimizer experiment"""
    set_seed(seed)
    
    # Create model and apply initialization
    model = SimpleCNN(num_classes=10).to(device)
    apply_custom_initialization(model, init_method)
    
    # FAIRNESS FIX: Optimizer-specific learning rates
    # 
    # PROBLEM: Using lr=0.001 for all optimizers is UNFAIR to SGD.
    # On MNIST/ConvNets, SGD needs lr=0.01-0.1 to converge well,
    # while Adam works great at lr=0.001 due to adaptive scaling.
    #
    # Using the same lr=0.001 for both makes SGD appear "sensitive to initialization"
    # when it's actually just starved of learning rate.
    #
    # SOLUTION: Use optimizer-preferred base LRs from published defaults.
    # This isolates the effect of INITIALIZATION from learning rate mismatch.
    
    # Create optimizer with fair LR per optimizer type
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD needs 10x more
    elif optimizer_name == 'SGD_Momentum':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam default
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
        
        # Check for divergence (NaN or very large loss)
        if np.isnan(train_loss) or train_loss > 100:
            logging.warning(f"Training diverged at epoch {epoch+1}: {init_method} + {optimizer_name}")
            break
    
    training_time = time.time() - start_time
    
    # Analyze convergence
    final_test_acc = history[-1]['test_acc']
    best_test_acc = max(h['test_acc'] for h in history)
    
    # Find epoch where accuracy reaches 90% of final (convergence speed)
    target_acc = 0.9 * final_test_acc
    convergence_epoch = None
    for h in history:
        if h['test_acc'] >= target_acc:
            convergence_epoch = h['epoch']
            break
    
    return {
        'init_method': init_method,
        'optimizer': optimizer_name,
        'final_test_acc': final_test_acc,
        'best_test_acc': best_test_acc,
        'convergence_epoch': convergence_epoch if convergence_epoch else epochs,
        'training_time': training_time,
        'diverged': np.isnan(history[-1]['train_loss']),
        'seed': seed,
        'history': history
    }


def run_initialization_ablation(
    results_dir: str = "results/initialization_ablation",
    seeds: List[int] = [1, 2, 3, 4, 5],
    epochs: int = 10,
    quick: bool = False
) -> pd.DataFrame:
    """
    Run comprehensive initialization-optimizer ablation study.
    
    Args:
        results_dir: Directory to save results
        seeds: List of random seeds
        epochs: Number of training epochs
        quick: If True, use fewer configurations for testing
    
    Returns:
        DataFrame with aggregated results
    """
    print("="*80)
    print("OPTIMIZER-INITIALIZATION INTERACTION ABLATION STUDY")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    
    # Setup data loaders
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform_test
    )
    
    if quick:
        train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
        test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
    
    train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=42, num_workers=2, pin_memory=True)
    test_loader = make_dataloader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    # Define configurations
    if quick:
        init_methods = ['normal_small', 'xavier_normal', 'kaiming_normal']
        optimizers = ['SGD', 'Adam']
    else:
        init_methods = [
            'uniform_small',
            'normal_small',
            'xavier_uniform',
            'xavier_normal',
            'kaiming_uniform',
            'kaiming_normal'
        ]
        optimizers = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW']
    
    # Run experiments
    all_results = []
    
    for init_method in init_methods:
        for optimizer_name in optimizers:
            print(f"\n{'='*80}")
            print(f"Config: {init_method} + {optimizer_name}")
            print(f"{'='*80}")
            
            config_results = []
            
            for seed in seeds:
                print(f"  Seed {seed}...", end=" ")
                result = run_single_experiment(
                    init_method, optimizer_name,
                    train_loader, test_loader,
                    device, epochs=epochs, seed=seed
                )
                config_results.append(result)
                print(f"Acc: {result['final_test_acc']:.2f}%, Converged: {result['convergence_epoch']}")
            
            # Aggregate results across seeds
            test_accs = [r['final_test_acc'] for r in config_results]
            conv_epochs = [r['convergence_epoch'] for r in config_results]
            diverged_count = sum(1 for r in config_results if r['diverged'])
            
            all_results.append({
                'initialization': init_method,
                'optimizer': optimizer_name,
                'mean_test_acc': np.mean(test_accs),
                'std_test_acc': np.std(test_accs),
                'mean_convergence_epoch': np.mean(conv_epochs),
                'std_convergence_epoch': np.std(conv_epochs),
                'divergence_rate': diverged_count / len(seeds),
                'n_seeds': len(seeds)
            })
            
            print(f"  Summary: {np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}%, " 
                  f"Converge: {np.mean(conv_epochs):.1f} epochs")
    
    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(all_results)
    df.to_csv(results_path / "initialization_ablation_summary.csv", index=False)
    
    # Generate visualizations
    try:
        # Prepare data for visualization (pivot to get configuration column)
        viz_df = df.copy()
        viz_df['configuration'] = viz_df['optimizer'] + ' + ' + viz_df['initialization']
        
        features = ['Xavier', 'Kaiming', 'Uniform', 'Normal']  # Initialization types
        generate_all_ablation_plots(
            df=viz_df,
            results_dir=results_dir,
            study_name='initialization_ablation',
            group_col='configuration',
            value_col='mean_test_acc',
            baseline_name='SGD + Xavier',  # Use a reasonable baseline
            features=features
        )
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_path / 'initialization_ablation_summary.csv'}")
    print(f"Visualizations saved to: {results_path / 'visualizations/'}")
    
    # Analysis: Which optimizer is most robust to initialization?
    print(f"\n{'='*80}")
    print("ANALYSIS: Optimizer Robustness to Initialization")
    print(f"{'='*80}")
    
    # Group by optimizer and compute variance across initializations
    robustness = df.groupby('optimizer').agg({
        'mean_test_acc': ['mean', 'std'],
        'std_test_acc': 'mean'
    }).round(2)
    
    print("\nAverage performance across all initializations:")
    print(robustness)
    
    # Find best initialization for each optimizer
    print(f"\n{'='*80}")
    print("BEST INITIALIZATION FOR EACH OPTIMIZER")
    print(f"{'='*80}")
    
    from src.utils.type_guards import ensure_series, ensure_dataframe
    for opt in ensure_series(df['optimizer']).unique():
        opt_df = ensure_dataframe(df[df['optimizer'] == opt])
        best_init = opt_df.loc[ensure_series(opt_df['mean_test_acc']).idxmax()]
        print(f"\n{opt}:")
        print(f"  Best init: {best_init['initialization']}")
        print(f"  Accuracy: {best_init['mean_test_acc']:.2f} ± {best_init['std_test_acc']:.2f}%")
        print(f"  Convergence: {best_init['mean_convergence_epoch']:.1f} epochs")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimizer-Initialization Ablation Study')
    parser.add_argument('--results-dir', type=str, default='results/initialization_ablation',
                        help='Directory to save results')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5',
                        help='Comma-separated list of random seeds')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with fewer configurations')
    
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    run_initialization_ablation(
        results_dir=args.results_dir,
        seeds=seeds,
        epochs=args.epochs,
        quick=args.quick
    )


if __name__ == '__main__':
    main()
