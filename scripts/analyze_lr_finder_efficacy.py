#!/usr/bin/env python3
"""
LR Finder Efficacy Study
Compares convergence with default LR (0.001) vs Auto-Tuned LR from LRFinder.

This addresses Phase 4.1 recommendation from the audit report.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import SimpleMLP
from src.core.training_enhancements import LRFinder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_mnist_data(batch_size=128):
    """Load MNIST dataset."""
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / max(1, len(train_loader)), 100. * correct / max(1, total)


def evaluate(model, test_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / max(1, len(test_loader)), 100. * correct / max(1, total)


def run_training(lr, epochs, device, seed=42):
    """Run training with specified learning rate."""
    torch.manual_seed(seed)
    
    # Load data
    train_loader, test_loader = load_mnist_data()
    
    # Create model
    model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        logging.info(f"Epoch {epoch+1}/{epochs} - LR={lr:.6f} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return history, test_acc


def find_optimal_lr_wrapper(device, seed=42):
    """Find optimal learning rate using LRFinder."""
    torch.manual_seed(seed)
    
    # Load data
    train_loader, _ = load_mnist_data()
    
    # Create model
    model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Run LR Finder
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    try:
        # Run LR range test (no input_transform parameter)
        lrs, losses = lr_finder.range_test(
            train_loader,
            start_lr=1e-5,
            end_lr=1.0,
            num_iter=100,
            verbose=False
        )
        # Get suggested LR from the LRFinder
        suggested_lr = lr_finder.suggest_lr()
        if suggested_lr is None or suggested_lr <= 0:
            logging.warning("LR Finder returned invalid LR. Using default LR=0.001")
            return 0.001
        logging.info(f"✅ LR Finder suggested LR: {suggested_lr:.6f}")
        return suggested_lr
    except Exception as e:
        logging.warning(f"LR Finder failed: {e}. Using default LR=0.001")
        return 0.001


def compare_lr_finder_vs_default(epochs=20, seeds=[1, 2, 3], output_dir='results/lr_finder_efficacy'):
    """
    Compare convergence with default LR (0.001) vs Auto-Tuned LR.
    
    This is the CRITICAL function addressing Phase 4.1 recommendation.
    """
    logging.info("="*80)
    logging.info("LR FINDER EFFICACY STUDY")
    logging.info("="*80)
    logging.info(f"Comparing Default LR (0.001) vs Auto-Tuned LR")
    logging.info(f"Epochs: {epochs}, Seeds: {seeds}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for seed in seeds:
        logging.info(f"\n{'='*80}")
        logging.info(f"SEED {seed}")
        logging.info(f"{'='*80}")
        
        # Find optimal LR for this seed
        optimal_lr = find_optimal_lr_wrapper(device, seed)
        
        # Train with default LR
        logging.info(f"\n🔹 Training with DEFAULT LR = 0.001")
        default_history, default_final_acc = run_training(0.001, epochs, device, seed)
        
        # Train with optimal LR
        logging.info(f"\n🔹 Training with AUTO-TUNED LR = {optimal_lr:.6f}")
        optimal_history, optimal_final_acc = run_training(optimal_lr, epochs, device, seed)
        
        # Record results
        improvement = optimal_final_acc - default_final_acc
        results.append({
            'seed': seed,
            'default_lr': 0.001,
            'optimal_lr': optimal_lr,
            'default_final_test_acc': default_final_acc,
            'optimal_final_test_acc': optimal_final_acc,
            'improvement': improvement
        })
        
        logging.info(f"\n📊 SEED {seed} SUMMARY:")
        logging.info(f"   Default LR (0.001): {default_final_acc:.2f}% test accuracy")
        logging.info(f"   Auto-Tuned LR ({optimal_lr:.6f}): {optimal_final_acc:.2f}% test accuracy")
        logging.info(f"   Improvement: {improvement:+.2f}%")
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'lr_finder_efficacy_comparison.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"\nResults saved to {csv_path}")
    
    # Summary statistics
    logging.info(f"\n{'='*80}")
    logging.info("OVERALL SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"Mean Default LR Test Accuracy: {df['default_final_test_acc'].mean():.2f}% ± {df['default_final_test_acc'].std():.2f}%")
    logging.info(f"Mean Auto-Tuned LR Test Accuracy: {df['optimal_final_test_acc'].mean():.2f}% ± {df['optimal_final_test_acc'].std():.2f}%")
    logging.info(f"Mean Improvement: {df['improvement'].mean():+.2f}% ± {df['improvement'].std():.2f}%")
    
    # Generate plot
    plt.figure(figsize=(10, 6), dpi=300)
    x = range(len(seeds))
    plt.bar([i - 0.2 for i in x], df['default_final_test_acc'], width=0.4, label='Default LR (0.001)', alpha=0.8)
    plt.bar([i + 0.2 for i in x], df['optimal_final_test_acc'], width=0.4, label='Auto-Tuned LR', alpha=0.8)
    plt.xlabel('Seed')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('LR Finder Efficacy: Default vs Auto-Tuned Learning Rate')
    plt.xticks(x, seeds)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'lr_finder_efficacy_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"✅ Plot saved to {plot_path}")
    
    # Statistical test
    from scipy import stats
    if len(seeds) >= 3:
        t_stat, p_value = stats.ttest_rel(df['optimal_final_test_acc'], df['default_final_test_acc'])
        logging.info(f"\n📊 Paired t-test:")
        logging.info(f"   t-statistic: {t_stat:.4f}")
        logging.info(f"   p-value: {p_value:.4f}")
        if p_value < 0.05:
            logging.info(f"   ✅ SIGNIFICANT improvement with Auto-Tuned LR (p < 0.05)")
        else:
            logging.info(f"   ⚠️  No significant difference (p >= 0.05)")
    
    logging.info(f"\n{'='*80}")
    logging.info("CONCLUSION:")
    if df['improvement'].mean() > 0:
        logging.info(f"✅ LR Finder provides {df['improvement'].mean():.2f}% average improvement")
        logging.info("   RECOMMENDATION: Enable --auto-lr for production runs")
    else:
        logging.info(f"⚠️  LR Finder shows {df['improvement'].mean():.2f}% average change")
        logging.info("   RECOMMENDATION: Default LR (0.001) is adequate for MNIST/Adam")
    logging.info(f"{'='*80}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LR Finder Efficacy Study')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--seeds', type=str, default='1,2,3', help='Comma-separated seeds')
    parser.add_argument('--output-dir', type=str, default='results/lr_finder_efficacy',
                       help='Output directory')
    
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    compare_lr_finder_vs_default(
        epochs=args.epochs,
        seeds=seeds,
        output_dir=args.output_dir
    )
