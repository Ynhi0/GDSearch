#!/usr/bin/env python3
"""
Additional Ablation Studies for Academic Completeness

This module provides ablation studies for hyperparameters and techniques
that are commonly used but not yet systematically analyzed:

1. Gradient Clipping - impact on training stability and convergence
2. Label Smoothing - effect on generalization and overconfidence
3. Data Augmentation - contribution to test accuracy
4. Model Architecture - hidden layer size sensitivity
5. Dropout - regularization vs performance tradeoff

REFACTORED: Now imports all components from src.core to eliminate code duplication.

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
from src.utils.constants import MNIST_MEAN, MNIST_STD
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import os
from tqdm import tqdm

# Import from core library (eliminates code duplication)
from src.core.models import SimpleMLP
from src.core.training_utils import LabelSmoothingCrossEntropy
from src.core.data_utils import get_mnist_loaders
from src.utils.plot_helpers import arr_to_numpy_float


def load_mnist_with_augmentation(
    augmentation: bool = False,
    batch_size: int = 128,
    quick: bool = False,
    seed: int = 42
) -> tuple:
    """
    Load MNIST with optional data augmentation.

    REFACTORED: Now uses get_mnist_loaders from core with custom transforms.

    Args:
        augmentation: Whether to apply data augmentation
        batch_size: Batch size for DataLoader
        quick: If True, use subset for fast testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, test_loader)
    """
    if augmentation:
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

    # Use core data_utils function for consistent loading
    data_root = './data'
    train_dataset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform_test
    )

    if quick:
        train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
        test_dataset = torch.utils.data.Subset(test_dataset, range(2000))

    # Use core dataloader utilities
    from src.core.dataloader_utils import make_dataloader
    train_loader = make_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True,
        seed=seed, num_workers=2, pin_memory=True
    )
    test_loader = make_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train_and_evaluate_with_clipping(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    device,
    gradient_clip=None
):
    """Train model with optional gradient clipping and return final metrics"""
    model.to(device)

    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Apply gradient clipping if specified
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
            
        # Evaluate at end of epoch
        model.eval()
        train_acc, train_avg_loss = evaluate(model, train_loader, nn.CrossEntropyLoss(), device)
        test_acc, test_avg_loss = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_avg_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_avg_loss:.4f}, Acc: {test_acc:.2f}%", flush=True)

        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': train_avg_loss,
            'test_loss': test_avg_loss
        })

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_loss': train_avg_loss,
        'test_loss': test_avg_loss
    }, history


def evaluate(model, loader, criterion, device):
    """Evaluate model accuracy and loss"""
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            output = model(data)
            loss = criterion(output, target)
            # BUG FIX: Weight loss by batch size for correct averaging
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / max(1, total)
    # BUG FIX: Divide by total samples, not number of batches
    avg_loss = total_loss / max(1, total)
    return accuracy, avg_loss


def run_gradient_clipping_ablation(
    clip_values: Optional[List[Optional[float]]] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    lr: float = 0.01,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/ablations'
) -> pd.DataFrame:
    """
    Ablation study: Impact of gradient clipping on training

    Tests whether gradient clipping improves stability and convergence
    """
    if clip_values is None:
        clip_values = [None, 0.5, 1.0, 5.0, 10.0]  # None = no clipping

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 GRADIENT CLIPPING ABLATION STUDY")
    print("=" * 80)
    print(f"Clip values: {clip_values}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print("=" * 80)

    results = []

    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for clip_val in tqdm(clip_values, desc="  Gradient clip values"):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load data
            train_loader, test_loader = load_mnist_with_augmentation(
                augmentation=False, batch_size=128, quick=quick
            )

            # Create model
            model = SimpleMLP()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # Train
            metrics, history = train_and_evaluate_with_clipping(
                model, train_loader, test_loader, optimizer, criterion,
                epochs, device, gradient_clip=clip_val
            )

            results.append({
                'Seed': seed,
                'Gradient_Clip': str(clip_val) if clip_val is not None else 'None',
                'Clip_Value': clip_val if clip_val is not None else 0.0,
                'Train_Acc': metrics['train_acc'],
                'Test_Acc': metrics['test_acc'],
                'Train_Loss': metrics['train_loss'],
                'Test_Loss': metrics['test_loss']
            })
            
            # Save per-run history for detailed analysis
            try:
                # Robustly import saving helper to avoid cyclical deps
                from run_all_kaggle import save_run_artifacts
                params = {'gradient_clip': clip_val, 'epochs': epochs, 'seed': seed}
                save_run_artifacts(output_dir, 'MNIST', 'SimpleMLP', f'GradientClip_{clip_val}', seed, history, params, device=device)
            except Exception as e:
                print(f"  Warning: Could not save per-run artifact: {e}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'gradient_clipping_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Create visualization
    create_ablation_plots(df, 'Gradient Clipping', output_dir, 'gradient_clipping')

    return df


def run_label_smoothing_ablation(
    smoothing_values: Optional[List[float]] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    lr: float = 0.01,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/ablations'
) -> pd.DataFrame:
    """
    Ablation study: Impact of label smoothing on generalization

    Tests whether label smoothing reduces overconfidence and improves test accuracy
    """
    if smoothing_values is None:
        smoothing_values = [0.0, 0.05, 0.1, 0.15, 0.2]

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 LABEL SMOOTHING ABLATION STUDY")
    print("=" * 80)
    print(f"Smoothing values: {smoothing_values}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print("=" * 80)

    results = []

    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for smoothing in tqdm(smoothing_values, desc="  Label smoothing values"):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load data
            train_loader, test_loader = load_mnist_with_augmentation(
                augmentation=False, batch_size=128, quick=quick
            )

            # Create model
            model = SimpleMLP()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            # Use label smoothing loss if smoothing > 0
            if smoothing > 0:
                criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
            else:
                criterion = nn.CrossEntropyLoss()

            # Train
            metrics, history = train_and_evaluate_with_clipping(
                model, train_loader, test_loader, optimizer, criterion,
                epochs, device, gradient_clip=None
            )

            results.append({
                'Seed': seed,
                'Label_Smoothing': smoothing,
                'Train_Acc': metrics['train_acc'],
                'Test_Acc': metrics['test_acc'],
                'Train_Loss': metrics['train_loss'],
                'Test_Loss': metrics['test_loss']
            })

            # Save per-run history
            try:
                from run_all_kaggle import save_run_artifacts
                params = {'label_smoothing': smoothing, 'epochs': epochs, 'seed': seed}
                save_run_artifacts(output_dir, 'MNIST', 'SimpleMLP', f'LabelSmoothing_{smoothing}', seed, history, params, device=device)
            except Exception as e:
                print(f"  Warning: Could not save per-run artifact: {e}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'label_smoothing_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Create visualization
    create_ablation_plots(df, 'Label Smoothing', output_dir, 'label_smoothing')

    return df


def run_data_augmentation_ablation(
    augmentation_configs: Optional[List[Dict]] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    lr: float = 0.01,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/ablations'
) -> pd.DataFrame:
    """
    Ablation study: Impact of data augmentation on performance

    Tests whether data augmentation improves generalization
    """
    if augmentation_configs is None:
        augmentation_configs = [
            {'name': 'None', 'use_aug': False},
            {'name': 'Standard', 'use_aug': True}
        ]

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 DATA AUGMENTATION ABLATION STUDY")
    print("=" * 80)
    print(f"Configs: {[c['name'] for c in augmentation_configs]}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print("=" * 80)

    results = []

    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for config in tqdm(augmentation_configs, desc="  Augmentation configs"):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load data
            train_loader, test_loader = load_mnist_with_augmentation(
                augmentation=config['use_aug'], batch_size=128, quick=quick
            )

            # Create model
            model = SimpleMLP()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # Train
            metrics, history = train_and_evaluate_with_clipping(
                model, train_loader, test_loader, optimizer, criterion,
                epochs, device, gradient_clip=None
            )

            results.append({
                'Seed': seed,
                'Augmentation': config['name'],
                'Train_Acc': metrics['train_acc'],
                'Test_Acc': metrics['test_acc'],
                'Train_Loss': metrics['train_loss'],
                'Test_Loss': metrics['test_loss']
            })

            # Save per-run history
            try:
                from run_all_kaggle import save_run_artifacts
                params = {'augmentation': config['name'], 'epochs': epochs, 'seed': seed}
                save_run_artifacts(output_dir, 'MNIST', 'SimpleMLP', f'Augmentation_{config["name"]}', seed, history, params, device=device)
            except Exception as e:
                print(f"  Warning: Could not save per-run artifact: {e}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'data_augmentation_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Create visualization
    create_categorical_ablation_plot(df, 'Data Augmentation', output_dir, 'data_augmentation')

    return df


def run_model_architecture_ablation(
    hidden_sizes: Optional[List[int]] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    lr: float = 0.01,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/ablations'
) -> pd.DataFrame:
    """
    Ablation study: Impact of hidden layer size on performance

    Tests model capacity vs generalization tradeoff
    """
    if hidden_sizes is None:
        hidden_sizes = [32, 64, 128, 256, 512]

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 MODEL ARCHITECTURE ABLATION STUDY")
    print("=" * 80)
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print("=" * 80)

    results = []

    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for hidden_size in tqdm(hidden_sizes, desc="  Hidden sizes"):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load data
            train_loader, test_loader = load_mnist_with_augmentation(
                augmentation=False, batch_size=128, quick=quick
            )

            # Create model with specific hidden size
            model = SimpleMLP(hidden_size=hidden_size)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())

            # Train
            metrics, history = train_and_evaluate_with_clipping(
                model, train_loader, test_loader, optimizer, criterion,
                epochs, device, gradient_clip=None
            )

            results.append({
                'Seed': seed,
                'Hidden_Size': hidden_size,
                'Num_Parameters': num_params,
                'Train_Acc': metrics['train_acc'],
                'Test_Acc': metrics['test_acc'],
                'Train_Loss': metrics['train_loss'],
                'Test_Loss': metrics['test_loss']
            })

            # Save per-run history
            try:
                from run_all_kaggle import save_run_artifacts
                params = {'hidden_size': hidden_size, 'epochs': epochs, 'seed': seed}
                save_run_artifacts(output_dir, 'MNIST', 'SimpleMLP', f'Arch_Hidden_{hidden_size}', seed, history, params, device=device)
            except Exception as e:
                print(f"  Warning: Could not save per-run artifact: {e}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'model_architecture_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Create visualization
    create_ablation_plots(df, 'Model Architecture (Hidden Size)', output_dir, 'model_architecture')

    return df


def run_dropout_ablation(
    dropout_rates: Optional[List[float]] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    lr: float = 0.01,
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/ablations'
) -> pd.DataFrame:
    """
    Ablation study: Impact of dropout regularization

    Tests dropout's effect on overfitting and generalization
    """
    if dropout_rates is None:
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 DROPOUT REGULARIZATION ABLATION STUDY")
    print("=" * 80)
    print(f"Dropout rates: {dropout_rates}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print("=" * 80)

    results = []

    for seed in seeds:
        print(f"\n📍 Seed {seed}")
        for dropout in tqdm(dropout_rates, desc="  Dropout rates"):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load data
            train_loader, test_loader = load_mnist_with_augmentation(
                augmentation=False, batch_size=128, quick=quick
            )

            # Create model with dropout
            model = SimpleMLP(dropout=dropout)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # Train
            metrics, history = train_and_evaluate_with_clipping(
                model, train_loader, test_loader, optimizer, criterion,
                epochs, device, gradient_clip=None
            )

            results.append({
                'Seed': seed,
                'Dropout_Rate': dropout,
                'Train_Acc': metrics['train_acc'],
                'Test_Acc': metrics['test_acc'],
                'Train_Loss': metrics['train_loss'],
                'Test_Loss': metrics['test_loss'],
                'Overfit_Gap': metrics['train_acc'] - metrics['test_acc']
            })

            # Save per-run history
            try:
                from run_all_kaggle import save_run_artifacts
                params = {'dropout': dropout, 'epochs': epochs, 'seed': seed}
                save_run_artifacts(output_dir, 'MNIST', 'SimpleMLP', f'Dropout_{dropout}', seed, history, params, device=device)
            except Exception as e:
                print(f"  Warning: Could not save per-run artifact: {e}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'dropout_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Create visualization
    create_ablation_plots(df, 'Dropout Regularization', output_dir, 'dropout')

    return df


def create_ablation_plots(df: pd.DataFrame, title: str, output_dir: str, filename: str):
    """Create standardized ablation study plots"""
    # Determine x-axis column.
    # Prefer numeric Clip_Value for gradient clipping to avoid string values like 'None'.
    candidate_cols = [c for c in df.columns if c != 'Seed']
    x_col = 'Clip_Value' if 'Clip_Value' in candidate_cols else candidate_cols[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title} Ablation Study on MNIST', fontsize=16, fontweight='bold')

    # 1. Test Accuracy
    ax = axes[0, 0]
    for seed in df['Seed'].unique():
        seed_data = df[df['Seed'] == seed]
        ax.plot(arr_to_numpy_float(seed_data[x_col]), arr_to_numpy_float(seed_data['Test_Acc']),
                marker='o', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel(x_col.replace('_', ' '), fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Train Accuracy
    ax = axes[0, 1]
    for seed in df['Seed'].unique():
        seed_data = df[df['Seed'] == seed]
        ax.plot(arr_to_numpy_float(seed_data[x_col]), arr_to_numpy_float(seed_data['Train_Acc']),
                marker='s', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel(x_col.replace('_', ' '), fontsize=12)
    ax.set_ylabel('Train Accuracy (%)', fontsize=12)
    ax.set_title('Train Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Test Loss
    ax = axes[1, 0]
    for seed in df['Seed'].unique():
        seed_data = df[df['Seed'] == seed]
        ax.plot(arr_to_numpy_float(seed_data[x_col]), arr_to_numpy_float(seed_data['Test_Loss']),
                marker='o', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel(x_col.replace('_', ' '), fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Overfitting Gap (if available)
    ax = axes[1, 1]
    if 'Overfit_Gap' in df.columns:
        for seed in df['Seed'].unique():
            seed_data = df[df['Seed'] == seed]
            ax.plot(arr_to_numpy_float(seed_data[x_col]), arr_to_numpy_float(seed_data['Overfit_Gap']),
                    marker='d', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel(x_col.replace('_', ' '), fontsize=12)
        ax.set_ylabel('Overfitting Gap (Train - Test Acc)', fontsize=12)
        ax.set_title('Overfitting Gap', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Summary statistics
        summary = df.groupby(x_col)[['Train_Acc', 'Test_Acc']].mean()
        summary.plot(kind='bar', ax=ax)
        ax.set_xlabel(x_col.replace('_', ' '), fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Summary: Train vs Test', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{filename}_ablation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {plot_path}")


def create_categorical_ablation_plot(df: pd.DataFrame, title: str, output_dir: str, filename: str):
    """Create plots for categorical ablation studies (e.g., augmentation on/off)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{title} Ablation Study on MNIST', fontsize=16, fontweight='bold')

    # 1. Test Accuracy Box Plot
    ax = axes[0]
    df.boxplot(column='Test_Acc', by='Augmentation', ax=ax)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Distribution', fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=0)

    # 2. Train vs Test comparison
    ax = axes[1]
    summary = df.groupby('Augmentation')[['Train_Acc', 'Test_Acc']].mean()
    summary.plot(kind='bar', ax=ax)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Train vs Test Accuracy', fontweight='bold')
    ax.legend(['Train', 'Test'])
    ax.grid(True, alpha=0.3)
    plt.sca(ax)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{filename}_ablation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {plot_path}")


def run_all_missing_ablations(
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    device: str = 'cpu',
    quick: bool = False,
    output_dir: str = 'results/missing_ablations'
) -> Dict[str, pd.DataFrame]:
    """Run all missing ablation studies"""
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    print("\n" + "=" * 80)
    print("🔬 RUNNING ALL MISSING ABLATION STUDIES")
    print("=" * 80)

    # 1. Gradient Clipping
    print("\n1️⃣  Gradient Clipping Ablation")
    results['gradient_clipping'] = run_gradient_clipping_ablation(
        epochs=epochs, seeds=seeds, device=device, quick=quick, output_dir=output_dir
    )

    # 2. Label Smoothing
    print("\n2️⃣  Label Smoothing Ablation")
    results['label_smoothing'] = run_label_smoothing_ablation(
        epochs=epochs, seeds=seeds, device=device, quick=quick, output_dir=output_dir
    )

    # 3. Data Augmentation
    print("\n3️⃣  Data Augmentation Ablation")
    results['data_augmentation'] = run_data_augmentation_ablation(
        epochs=epochs, seeds=seeds, device=device, quick=quick, output_dir=output_dir
    )

    # 4. Model Architecture
    print("\n4️⃣  Model Architecture Ablation")
    results['model_architecture'] = run_model_architecture_ablation(
        epochs=epochs, seeds=seeds, device=device, quick=quick, output_dir=output_dir
    )

    # 5. Dropout
    print("\n5️⃣  Dropout Regularization Ablation")
    results['dropout'] = run_dropout_ablation(
        epochs=epochs, seeds=seeds, device=device, quick=quick, output_dir=output_dir
    )

    print("\n" + "=" * 80)
    print("ALL MISSING ABLATION STUDIES COMPLETED")
    print("=" * 80)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Missing Ablation Studies')
    parser.add_argument('--ablation', type=str, default='all',
                       choices=['all', 'gradient_clip', 'label_smooth', 'augment', 'arch', 'dropout'],
                       help='Which ablation to run')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--output', type=str, default='results/missing_ablations', help='Output directory')

    args = parser.parse_args()

    if args.ablation == 'all':
        run_all_missing_ablations(
            epochs=args.epochs,
            device=args.device,
            quick=args.quick,
            output_dir=args.output
        )
    elif args.ablation == 'gradient_clip':
        run_gradient_clipping_ablation(
            epochs=args.epochs,
            device=args.device,
            quick=args.quick,
            output_dir=args.output
        )
    elif args.ablation == 'label_smooth':
        run_label_smoothing_ablation(
            epochs=args.epochs,
            device=args.device,
            quick=args.quick,
            output_dir=args.output
        )
    elif args.ablation == 'augment':
        run_data_augmentation_ablation(
            epochs=args.epochs,
            device=args.device,
            quick=args.quick,
            output_dir=args.output
        )
    elif args.ablation == 'arch':
        run_model_architecture_ablation(
            epochs=args.epochs,
            device=args.device,
            quick=args.quick,
            output_dir=args.output
        )
    elif args.ablation == 'dropout':
        run_dropout_ablation(
            epochs=args.epochs,
            device=args.device,
            quick=args.quick,
            output_dir=args.output
        )
