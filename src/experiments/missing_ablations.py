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
from typing import List, Dict, Optional
import os
from tqdm import tqdm


class SimpleMLP(nn.Module):
    """Configurable MLP for ablation studies"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10, dropout=0.0):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def load_mnist_with_augmentation(augmentation=False, batch_size=128, quick=False):
    """Load MNIST with optional data augmentation"""
    if augmentation:
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    if quick:
        train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
        test_dataset = torch.utils.data.Subset(test_dataset, range(2000))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    device,
    gradient_clip=None
):
    """Train model and return final metrics"""
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
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
    
    # Evaluate
    model.eval()
    train_acc, train_loss = evaluate(model, train_loader, nn.CrossEntropyLoss(), device)
    test_acc, test_loss = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss
    }


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
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss


def run_gradient_clipping_ablation(
    clip_values: List[float] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456],
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
            metrics = train_and_evaluate(
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
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'gradient_clipping_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
    
    # Create visualization
    create_ablation_plots(df, 'Gradient Clipping', output_dir, 'gradient_clipping')
    
    return df


def run_label_smoothing_ablation(
    smoothing_values: List[float] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456],
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
            metrics = train_and_evaluate(
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
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'label_smoothing_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
    
    # Create visualization
    create_ablation_plots(df, 'Label Smoothing', output_dir, 'label_smoothing')
    
    return df


def run_data_augmentation_ablation(
    augmentation_configs: List[Dict] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456],
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
            metrics = train_and_evaluate(
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
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'data_augmentation_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
    
    # Create visualization
    create_categorical_ablation_plot(df, 'Data Augmentation', output_dir, 'data_augmentation')
    
    return df


def run_model_architecture_ablation(
    hidden_sizes: List[int] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456],
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
            metrics = train_and_evaluate(
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
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'model_architecture_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
    
    # Create visualization
    create_ablation_plots(df, 'Model Architecture (Hidden Size)', output_dir, 'model_architecture')
    
    return df


def run_dropout_ablation(
    dropout_rates: List[float] = None,
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456],
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
            metrics = train_and_evaluate(
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
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'dropout_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
    
    # Create visualization
    create_ablation_plots(df, 'Dropout Regularization', output_dir, 'dropout')
    
    return df


def create_ablation_plots(df: pd.DataFrame, title: str, output_dir: str, filename: str):
    """Create standardized ablation study plots"""
    # Determine x-axis column (first column after 'Seed')
    x_col = [c for c in df.columns if c != 'Seed'][0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title} Ablation Study on MNIST', fontsize=16, fontweight='bold')
    
    # 1. Test Accuracy
    ax = axes[0, 0]
    for seed in df['Seed'].unique():
        seed_data = df[df['Seed'] == seed]
        ax.plot(seed_data[x_col], seed_data['Test_Acc'], 
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
        ax.plot(seed_data[x_col], seed_data['Train_Acc'], 
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
        ax.plot(seed_data[x_col], seed_data['Test_Loss'], 
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
            ax.plot(seed_data[x_col], seed_data['Overfit_Gap'], 
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
    
    print(f"✅ Plots saved to {plot_path}")


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
    
    print(f"✅ Plots saved to {plot_path}")


def run_all_missing_ablations(
    epochs: int = 10,
    seeds: List[int] = [42, 123, 456],
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
    print("✅ ALL MISSING ABLATION STUDIES COMPLETED")
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
