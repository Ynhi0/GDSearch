#!/usr/bin/env python3
"""
Beta Parameter Sensitivity Analysis on Real Neural Network Training

This module addresses the Vietnamese research proposal requirement:
"khảo sát hệ thống và trực quan hóa ảnh hưởng của các siêu tham số đặc trưng 
(β, β1, β2) lên các khía cạnh động học như quỹ đạo, tốc độ tức thời và độ ổn định"

Unlike hyperparameter_sensitivity.py (which runs on 2D functions), this module
runs β sweeps on REAL MNIST/CIFAR training to analyze:
- Impact on final accuracy
- Impact on convergence speed
- Impact on training dynamics (smoothness, oscillations)
- Impact on loss landscape navigation

Author: Research Team
Date: December 7, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import json
from tqdm import tqdm

# Import dynamics tracking
try:
    from src.core.dynamics_tracker import TrainingDynamicsTracker
    from src.analysis.dynamics_metrics import compute_instantaneous_speed, compute_smoothness_index
    HAS_DYNAMICS = True
except ImportError:
    HAS_DYNAMICS = False
    print("Dynamics tracking not available - metrics will be limited")


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def load_mnist(batch_size=128, quick=False):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    if quick:
        # Use subset for quick testing
        train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
        test_dataset = torch.utils.data.Subset(test_dataset, range(2000))
    
    train_loader = make_dataloader(train_dataset, batch_size=batch_size, shuffle=True, seed=42, num_workers=2, pin_memory=True)
    test_loader = make_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader


def train_with_beta(
    beta: float,
    optimizer_name: str,
    epochs: int = 20,
    lr: float = 0.01,
    device: str = 'cpu',
    track_dynamics: bool = True,
    quick: bool = False,
    seed: int = 42
) -> Dict:
    """
    Train MNIST with specific β value and track comprehensive metrics
    
    Args:
        beta: Momentum parameter (β for Momentum, β1 for Adam)
        optimizer_name: 'momentum' or 'adam'
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
        track_dynamics: Whether to track dynamics metrics
        quick: Use subset of data for quick testing
        seed: Random seed
    
    Returns:
        Dictionary with training history and dynamics metrics
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    train_loader, test_loader = load_mnist(quick=quick)
    
    # Create model
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer with specific beta
    if optimizer_name.lower() == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=beta)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta, 0.999))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Initialize dynamics tracker
    if track_dynamics and HAS_DYNAMICS:
        dynamics_tracker = TrainingDynamicsTracker()
        dynamics_tracker.set_initial_params(model)
    else:
        dynamics_tracker = None
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'grad_norm': [],
        'param_norm': [],
        'update_magnitude': []
    }
    
    # Store parameter snapshots for trajectory analysis
    param_snapshots = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        grad_norms = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Store old params
            old_params = [p.clone().detach() for p in model.parameters()]
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Compute gradient norm
            grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
            grad_norms.append(grad_norm.item())
            
            # Track dynamics BEFORE optimizer step
            if dynamics_tracker is not None:
                dynamics_tracker.track_step(
                    iteration=epoch * len(train_loader) + batch_idx,
                    loss=loss.item(),
                    model=model,
                    optimizer=optimizer
                )
            
            optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Compute param norm
        param_norm = torch.sqrt(sum(p.norm()**2 for p in model.parameters()))
        
        # Store snapshot
        param_snapshots.append(torch.cat([p.view(-1).clone().detach() for p in model.parameters()]).cpu().numpy())
        
        # Record history (validation only during training - use validation set for monitoring)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['grad_norm'].append(np.mean(grad_norms))
        history['param_norm'].append(param_norm.item())
        if len(param_snapshots) >= 2:
            update_mag = np.linalg.norm(param_snapshots[-1] - param_snapshots[-2])
            history['update_magnitude'].append(update_mag)
        else:
            history['update_magnitude'].append(0.0)
    
    # Final test evaluation (only after training completes - use test set only for final evaluation)
    logger.info("Evaluating final performance on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    final_test_acc = 100. * test_correct / test_total
    logger.info(f"Final Test Performance: Loss={test_loss:.4f}, Acc={final_test_acc:.2f}%")
    
    # Add final test metrics to history
    history['final_test_loss'] = test_loss
    history['final_test_acc'] = final_test_acc
    
    # Compute dynamics metrics
    dynamics_metrics = {}
    
    if dynamics_tracker is not None and len(dynamics_tracker.iterations) > 0:
        # Compute derived metrics
        dynamics_tracker.compute_derived_metrics()
        
        # Get summary
        dynamics_metrics['mean_grad_norm'] = np.mean(dynamics_tracker.grad_norms)
        dynamics_metrics['std_grad_norm'] = np.std(dynamics_tracker.grad_norms)
        dynamics_metrics['mean_update_mag'] = np.mean(dynamics_tracker.update_magnitudes)
        dynamics_metrics['final_param_distance'] = dynamics_tracker.param_distances[-1] if dynamics_tracker.param_distances else 0.0
    
    if len(param_snapshots) >= 3:
        param_array = np.array(param_snapshots)
        
        # Instantaneous speed
        speeds = []
        for i in range(1, len(param_array)):
            speed = np.linalg.norm(param_array[i] - param_array[i-1])
            speeds.append(speed)
        dynamics_metrics['mean_speed'] = np.mean(speeds)
        dynamics_metrics['std_speed'] = np.std(speeds)
        
        # Smoothness (angle changes)
        if HAS_DYNAMICS:
            smoothness = compute_smoothness_index(param_array)
            dynamics_metrics['smoothness'] = smoothness
        
        # Loss oscillations
        losses = np.array(history['train_loss'])
        ema_loss = pd.Series(losses).ewm(span=5).mean().values
        oscillations = losses - ema_loss
        dynamics_metrics['oscillation_index'] = np.std(oscillations)
        dynamics_metrics['final_loss_std'] = np.std(losses[-5:])  # Stability at end
    
    return {
        'beta': beta,
        'optimizer': optimizer_name,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'final_test_loss': history['test_loss'][-1],
        'history': history,
        'dynamics_metrics': dynamics_metrics,
        'param_snapshots': param_snapshots
    }


def run_momentum_beta_sensitivity(
    beta_values: List[float] = None,
    epochs: int = 20,
    seeds: List[int] = [42, 123, 456],
    lr: float = 0.01,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/beta_sensitivity'
) -> pd.DataFrame:
    """
    Run β sensitivity analysis for Momentum optimizer on MNIST
    
    This addresses the proposal requirement for β sensitivity on REAL training
    """
    if beta_values is None:
        beta_values = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🔬 MOMENTUM β SENSITIVITY ON MNIST TRAINING")
    print("=" * 80)
    print(f"β values: {beta_values}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 80)
    
    results = []
    
    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for beta in tqdm(beta_values, desc="  β sweep"):
            result = train_with_beta(
                beta=beta,
                optimizer_name='momentum',
                epochs=epochs,
                lr=lr,
                device=device,
                track_dynamics=True,
                quick=quick,
                seed=seed
            )
            result['seed'] = seed
            results.append(result)
    
    # Convert to DataFrame
    rows = []
    for r in results:
        row = {
            'Seed': r['seed'],
            'Beta': r['beta'],
            'Optimizer': r['optimizer'],
            'Final_Train_Acc': r['final_train_acc'],
            'Final_Test_Acc': r['final_test_acc'],
            'Final_Train_Loss': r['final_train_loss'],
            'Final_Test_Loss': r['final_test_loss']
        }
        # Add dynamics metrics
        for k, v in r['dynamics_metrics'].items():
            row[k] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save results
    csv_path = os.path.join(output_dir, 'momentum_beta_sensitivity_mnist.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Create visualizations
    create_beta_sensitivity_plots(df, output_dir, optimizer='Momentum')
    
    return df


def run_adam_beta_sensitivity(
    beta1_values: List[float] = None,
    beta2: float = 0.999,
    epochs: int = 20,
    seeds: List[int] = [42, 123, 456],
    lr: float = 0.001,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/beta_sensitivity'
) -> pd.DataFrame:
    """
    Run β1 sensitivity analysis for Adam optimizer on MNIST
    
    This addresses the proposal requirement for β1,β2 sensitivity on REAL training
    """
    if beta1_values is None:
        beta1_values = [0.5, 0.7, 0.9, 0.95, 0.99]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🔬 ADAM β1 SENSITIVITY ON MNIST TRAINING")
    print("=" * 80)
    print(f"β1 values: {beta1_values}")
    print(f"β2: {beta2} (fixed)")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 80)
    
    results = []
    
    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for beta1 in tqdm(beta1_values, desc="  β1 sweep"):
            result = train_with_beta(
                beta=beta1,  # β1 for Adam
                optimizer_name='adam',
                epochs=epochs,
                lr=lr,
                device=device,
                track_dynamics=True,
                quick=quick,
                seed=seed
            )
            result['seed'] = seed
            result['beta1'] = beta1
            result['beta2'] = beta2
            results.append(result)
    
    # Convert to DataFrame
    rows = []
    for r in results:
        row = {
            'Seed': r['seed'],
            'Beta1': r.get('beta1', r['beta']),
            'Beta2': r.get('beta2', beta2),
            'Optimizer': r['optimizer'],
            'Final_Train_Acc': r['final_train_acc'],
            'Final_Test_Acc': r['final_test_acc'],
            'Final_Train_Loss': r['final_train_loss'],
            'Final_Test_Loss': r['final_test_loss']
        }
        # Add dynamics metrics
        for k, v in r['dynamics_metrics'].items():
            row[k] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save results
    csv_path = os.path.join(output_dir, 'adam_beta_sensitivity_mnist.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Create visualizations
    create_beta_sensitivity_plots(df, output_dir, optimizer='Adam')
    
    return df


def run_adam_beta2_sensitivity(
    beta1: float = 0.9,
    beta2_values: List[float] = None,
    epochs: int = 20,
    seeds: List[int] = [42, 123, 456],
    lr: float = 0.001,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/beta_sensitivity'
) -> pd.DataFrame:
    """
    Run β2 sensitivity analysis for Adam optimizer on MNIST
    
    Addresses proposal requirement: full (β1, β2) hyperparameter analysis
    """
    if beta2_values is None:
        beta2_values = [0.9, 0.95, 0.99, 0.999, 0.9999]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🔬 ADAM β2 SENSITIVITY ON MNIST TRAINING")
    print("=" * 80)
    print(f"β1: {beta1} (fixed)")
    print(f"β2 values: {beta2_values}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 80)
    
    results = []
    
    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for beta2 in tqdm(beta2_values, desc="  β2 sweep"):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Load data
            train_loader, test_loader = load_mnist(batch_size=128, quick=quick)
            
            # Create model
            model = SimpleMLP().to(device)
            
            # Adam with specific β1, β2
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
            criterion = nn.CrossEntropyLoss()
            
            # Track dynamics
            if HAS_DYNAMICS:
                dynamics_tracker = TrainingDynamicsTracker()
                dynamics_tracker.set_initial_params(model)
            
            # Training
            for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    # Track dynamics
                    if HAS_DYNAMICS and batch_idx % 10 == 0:
                        dynamics_tracker.track_step(
                            iteration=epoch * len(train_loader) + batch_idx,
                            loss=loss.item(),
                            model=model,
                            optimizer=optimizer
                        )
            
            # Evaluate
            model.eval()
            train_acc, train_loss = evaluate(model, train_loader, criterion, device)
            test_acc, test_loss = evaluate(model, test_loader, criterion, device)
            
            # Compute dynamics metrics
            dynamics_metrics = {}
            if HAS_DYNAMICS:
                if len(dynamics_tracker.iterations) > 0:
                    dynamics_metrics = dynamics_tracker.compute_derived_metrics()
            
            result = {
                'seed': seed,
                'beta1': beta1,
                'beta2': beta2,
                'optimizer': 'Adam',
                'final_train_acc': train_acc,
                'final_test_acc': test_acc,
                'final_train_loss': train_loss,
                'final_test_loss': test_loss,
                'dynamics_metrics': dynamics_metrics
            }
            results.append(result)
    
    # Convert to DataFrame
    rows = []
    for r in results:
        row = {
            'Seed': r['seed'],
            'Beta1': r['beta1'],
            'Beta2': r['beta2'],
            'Optimizer': r['optimizer'],
            'Final_Train_Acc': r['final_train_acc'],
            'Final_Test_Acc': r['final_test_acc'],
            'Final_Train_Loss': r['final_train_loss'],
            'Final_Test_Loss': r['final_test_loss']
        }
        # Add dynamics metrics
        for k, v in r['dynamics_metrics'].items():
            row[k] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save results
    csv_path = os.path.join(output_dir, 'adam_beta2_sensitivity_mnist.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Create visualizations (reuse with Beta2 column)
    create_beta2_sensitivity_plots(df, output_dir)
    
    return df


def run_adam_beta1_beta2_grid(
    beta1_values: List[float] = None,
    beta2_values: List[float] = None,
    epochs: int = 15,
    seeds: List[int] = [42, 123],
    lr: float = 0.001,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/beta_sensitivity'
) -> pd.DataFrame:
    """
    Run joint (β1, β2) grid search for Adam optimizer on MNIST
    
    This provides comprehensive understanding of how β1 and β2 interact
    """
    if beta1_values is None:
        beta1_values = [0.7, 0.9, 0.99]
    if beta2_values is None:
        beta2_values = [0.9, 0.99, 0.999]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🔬 ADAM (β1, β2) GRID SEARCH ON MNIST TRAINING")
    print("=" * 80)
    print(f"β1 values: {beta1_values}")
    print(f"β2 values: {beta2_values}")
    print(f"Grid size: {len(beta1_values)} × {len(beta2_values)} = {len(beta1_values) * len(beta2_values)}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 80)
    
    results = []
    total_runs = len(beta1_values) * len(beta2_values) * len(seeds)
    run_count = 0
    
    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for beta1 in beta1_values:
            for beta2 in beta2_values:
                run_count += 1
                print(f"  [{run_count}/{total_runs}] β1={beta1:.3f}, β2={beta2:.4f}")
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Load data
                train_loader, test_loader = load_mnist(batch_size=128, quick=quick)
                
                # Create model
                model = SimpleMLP().to(device)
                
                # Adam with specific β1, β2
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
                criterion = nn.CrossEntropyLoss()
                
                # Track dynamics
                if HAS_DYNAMICS:
                    dynamics_tracker = TrainingDynamicsTracker()
                    dynamics_tracker.set_initial_params(model)
                
                # Training
                for epoch in range(epochs):
                    model.train()
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        
                        # Track dynamics
                        if HAS_DYNAMICS and batch_idx % 10 == 0:
                            dynamics_tracker.track_step(
                                iteration=epoch * len(train_loader) + batch_idx,
                                loss=loss.item(),
                                model=model,
                                optimizer=optimizer
                            )
                
                # Evaluate
                model.eval()
                train_acc, train_loss = evaluate(model, train_loader, criterion, device)
                test_acc, test_loss = evaluate(model, test_loader, criterion, device)
                
                # Compute dynamics metrics
                dynamics_metrics = {}
                if HAS_DYNAMICS:
                    if len(dynamics_tracker.iterations) > 0:
                        dynamics_metrics = dynamics_tracker.compute_derived_metrics()
                
                result = {
                    'seed': seed,
                    'beta1': beta1,
                    'beta2': beta2,
                    'optimizer': 'Adam',
                    'final_train_acc': train_acc,
                    'final_test_acc': test_acc,
                    'final_train_loss': train_loss,
                    'final_test_loss': test_loss,
                    'dynamics_metrics': dynamics_metrics
                }
                results.append(result)
    
    # Convert to DataFrame
    rows = []
    for r in results:
        row = {
            'Seed': r['seed'],
            'Beta1': r['beta1'],
            'Beta2': r['beta2'],
            'Optimizer': r['optimizer'],
            'Final_Train_Acc': r['final_train_acc'],
            'Final_Test_Acc': r['final_test_acc'],
            'Final_Train_Loss': r['final_train_loss'],
            'Final_Test_Loss': r['final_test_loss']
        }
        # Add dynamics metrics
        for k, v in r['dynamics_metrics'].items():
            row[k] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save results
    csv_path = os.path.join(output_dir, 'adam_beta1_beta2_grid_mnist.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Create 2D heatmap visualizations
    create_beta_grid_plots(df, output_dir)
    
    return df


def evaluate(model, loader, criterion, device):
    """Evaluate model accuracy and loss"""
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100.0 * correct / max(1, total)
    avg_loss = total_loss / max(1, len(loader))
    return accuracy, avg_loss


def create_beta2_sensitivity_plots(df: pd.DataFrame, output_dir: str):
    """Create visualizations for β2 sensitivity"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Adam β2 Sensitivity on MNIST Training (Real Neural Network)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Test Accuracy vs β2
    ax = axes[0, 0]
    for seed in df['Seed'].unique():
        seed_data = df[df['Seed'] == seed]
        ax.plot(seed_data['Beta2'], seed_data['Final_Test_Acc'], 
                marker='o', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel('β2', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy vs β2', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. Train Loss vs β2
    ax = axes[0, 1]
    for seed in df['Seed'].unique():
        seed_data = df[df['Seed'] == seed]
        ax.plot(seed_data['Beta2'], seed_data['Final_Train_Loss'], 
                marker='o', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel('β2', fontsize=12)
    ax.set_ylabel('Final Train Loss', fontsize=12)
    ax.set_title('Train Loss vs β2', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 3. Smoothness vs β2
    ax = axes[1, 0]
    if 'smoothness' in df.columns:
        for seed in df['Seed'].unique():
            seed_data = df[df['Seed'] == seed]
            ax.plot(seed_data['Beta2'], seed_data['smoothness'], 
                    marker='o', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel('β2', fontsize=12)
        ax.set_ylabel('Smoothness Index', fontsize=12)
        ax.set_title('Dynamics: Smoothness vs β2', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # 4. Oscillation vs β2
    ax = axes[1, 1]
    if 'oscillation_index' in df.columns:
        for seed in df['Seed'].unique():
            seed_data = df[df['Seed'] == seed]
            ax.plot(seed_data['Beta2'], seed_data['oscillation_index'], 
                    marker='o', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel('β2', fontsize=12)
        ax.set_ylabel('Oscillation Index', fontsize=12)
        ax.set_title('Dynamics: Oscillations vs β2', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'adam_beta2_sensitivity_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"β2 visualizations saved to {plot_path}")


def create_beta_grid_plots(df: pd.DataFrame, output_dir: str):
    """Create 2D heatmap visualizations for (β1, β2) grid search"""
    
    # Aggregate across seeds
    agg_df = df.groupby(['Beta1', 'Beta2']).agg({
        'Final_Test_Acc': 'mean',
        'Final_Train_Loss': 'mean',
        'mean_speed': 'mean' if 'mean_speed' in df.columns else lambda x: 0,
        'oscillation_index': 'mean' if 'oscillation_index' in df.columns else lambda x: 0
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Adam (β1, β2) Grid Search on MNIST Training', 
                 fontsize=16, fontweight='bold')
    
    # 1. Test Accuracy Heatmap
    ax = axes[0, 0]
    pivot = agg_df.pivot(index='Beta2', columns='Beta1', values='Final_Test_Acc')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Test Acc (%)'})
    ax.set_title('Final Test Accuracy', fontweight='bold')
    ax.set_xlabel('β1')
    ax.set_ylabel('β2')
    
    # 2. Train Loss Heatmap
    ax = axes[0, 1]
    pivot = agg_df.pivot(index='Beta2', columns='Beta1', values='Final_Train_Loss')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Train Loss'})
    ax.set_title('Final Train Loss', fontweight='bold')
    ax.set_xlabel('β1')
    ax.set_ylabel('β2')
    
    # 3. Mean Speed Heatmap
    ax = axes[1, 0]
    if 'mean_speed' in agg_df.columns and agg_df['mean_speed'].sum() > 0:
        pivot = agg_df.pivot(index='Beta2', columns='Beta1', values='mean_speed')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label': 'Speed'})
        ax.set_title('Mean Update Speed', fontweight='bold')
        ax.set_xlabel('β1')
        ax.set_ylabel('β2')
    
    # 4. Oscillation Heatmap
    ax = axes[1, 1]
    if 'oscillation_index' in agg_df.columns and agg_df['oscillation_index'].sum() > 0:
        pivot = agg_df.pivot(index='Beta2', columns='Beta1', values='oscillation_index')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Oscillation'})
        ax.set_title('Oscillation Index', fontweight='bold')
        ax.set_xlabel('β1')
        ax.set_ylabel('β2')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'adam_beta1_beta2_grid_heatmaps.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grid search heatmaps saved to {plot_path}")


def create_beta_sensitivity_plots(df: pd.DataFrame, output_dir: str, optimizer: str):
    """Create comprehensive visualizations for β sensitivity"""
    
    beta_col = 'Beta' if optimizer == 'Momentum' else 'Beta1'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{optimizer} β Sensitivity on MNIST Training (Real Neural Network)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Final Test Accuracy vs β
    ax = axes[0, 0]
    for seed in df['Seed'].unique():
        seed_data = df[df['Seed'] == seed]
        ax.plot(seed_data[beta_col], seed_data['Final_Test_Acc'], 
                marker='o', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel(f'{beta_col}', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy vs β', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final Train Loss vs β
    ax = axes[0, 1]
    for seed in df['Seed'].unique():
        seed_data = df[df['Seed'] == seed]
        ax.plot(seed_data[beta_col], seed_data['Final_Train_Loss'], 
                marker='o', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel(f'{beta_col}', fontsize=12)
    ax.set_ylabel('Final Train Loss', fontsize=12)
    ax.set_title('Train Loss vs β', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Mean Speed vs β (Dynamics)
    ax = axes[0, 2]
    if 'mean_speed' in df.columns:
        for seed in df['Seed'].unique():
            seed_data = df[df['Seed'] == seed]
            ax.plot(seed_data[beta_col], seed_data['mean_speed'], 
                    marker='o', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel(f'{beta_col}', fontsize=12)
        ax.set_ylabel('Mean Update Speed', fontsize=12)
        ax.set_title('Dynamics: Speed vs β', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Smoothness vs β
    ax = axes[1, 0]
    if 'smoothness' in df.columns:
        for seed in df['Seed'].unique():
            seed_data = df[df['Seed'] == seed]
            ax.plot(seed_data[beta_col], seed_data['smoothness'], 
                    marker='o', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel(f'{beta_col}', fontsize=12)
        ax.set_ylabel('Smoothness Index', fontsize=12)
        ax.set_title('Dynamics: Smoothness vs β', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Oscillation Index vs β
    ax = axes[1, 1]
    if 'oscillation_index' in df.columns:
        for seed in df['Seed'].unique():
            seed_data = df[df['Seed'] == seed]
            ax.plot(seed_data[beta_col], seed_data['oscillation_index'], 
                    marker='o', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel(f'{beta_col}', fontsize=12)
        ax.set_ylabel('Oscillation Index', fontsize=12)
        ax.set_title('Dynamics: Oscillations vs β', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Summary Heatmap
    ax = axes[1, 2]
    if len(df[beta_col].unique()) > 1:
        # Aggregate across seeds
        agg_df = df.groupby(beta_col)[['Final_Test_Acc', 'mean_speed', 'oscillation_index']].mean()
        # Normalize for heatmap
        agg_norm = (agg_df - agg_df.min()) / (agg_df.max() - agg_df.min())
        sns.heatmap(agg_norm.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Normalized Value'})
        ax.set_xlabel(f'{beta_col}', fontsize=12)
        ax.set_title('Normalized Metrics Heatmap', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{optimizer.lower()}_beta_sensitivity_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {plot_path}")


def main():
    """Main function for testing"""
    import argparse
    parser = argparse.ArgumentParser(description='β Sensitivity Analysis on Real Training')
    parser.add_argument('--optimizer', type=str, default='momentum', 
                       choices=['momentum', 'adam', 'adam_beta2', 'adam_grid'],
                       help='Optimizer to analyze')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test with subset')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output', type=str, default='results/beta_sensitivity',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.optimizer == 'momentum':
        df = run_momentum_beta_sensitivity(
            epochs=args.epochs,
            quick=args.quick,
            device=args.device,
            output_dir=args.output
        )
        beta_col = 'Beta'
    elif args.optimizer == 'adam':
        df = run_adam_beta_sensitivity(
            epochs=args.epochs,
            quick=args.quick,
            device=args.device,
            output_dir=args.output
        )
        beta_col = 'Beta1'
    elif args.optimizer == 'adam_beta2':
        df = run_adam_beta2_sensitivity(
            epochs=args.epochs,
            quick=args.quick,
            device=args.device,
            output_dir=args.output
        )
        beta_col = 'Beta2'
    else:  # adam_grid
        df = run_adam_beta1_beta2_grid(
            epochs=args.epochs,
            quick=args.quick,
            device=args.device,
            output_dir=args.output
        )
        beta_col = 'Beta1'
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    if args.optimizer == 'adam_grid':
        print(df.groupby(['Beta1', 'Beta2'])[
            ['Final_Test_Acc', 'Final_Train_Loss', 'mean_speed', 'oscillation_index']
        ].mean())
    else:
        print(df.groupby(beta_col)[
            ['Final_Test_Acc', 'Final_Train_Loss', 'mean_speed', 'oscillation_index']
        ].mean())


if __name__ == '__main__':
    main()
