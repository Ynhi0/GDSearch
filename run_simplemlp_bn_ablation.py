#!/usr/bin/env python3
"""
SimpleMLP Batch Normalization Ablation Study
Tests optimizer performance with/without BN to isolate optimizer effects

INTEGRATION FIX (Issue #26): Addresses "Inconsistent Architecture Normalization"
from QA Report. Tests SimpleMLP with/without BN vs Adam/SGD to separate
optimizer effects from normalization effects.

Usage:
    python run_simplemlp_bn_ablation.py --seeds 1,2,3
    python run_simplemlp_bn_ablation.py --quick  # Fast test
"""

import sys
import argparse
import logging
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import pandas as pd

# Setup reproducibility FIRST
from src.utils.reproducibility import setup_experiment_reproducibility
setup_experiment_reproducibility(seed=42, deterministic=False)

from src.core.models import SimpleMLP
from src.core.data_utils import get_mnist_loaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return total_loss / len(loader), correct / total


def run_ablation(use_bn, optimizer_name, lr, seed, epochs, device):
    """Run single ablation experiment"""
    setup_experiment_reproducibility(seed=seed)

    # Load data
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=128,
        seed=seed,
        val_split=0.1
    )

    # Create model
    model = SimpleMLP(
        input_size=784,
        hidden_size=256,
        num_classes=10,
        dropout=0.0,
        use_bn=use_bn
    ).to(device)

    # Create optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif optimizer_name == 'SGD_Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Training loop
    history = []
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}: "
                        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    elapsed = time.time() - start_time

    # Final metrics (guard against empty history)
    if not history:
        return {
            'use_bn': use_bn,
            'optimizer': optimizer_name,
            'lr': lr,
            'seed': seed,
            'final_train_acc': float('nan'),
            'final_val_acc': float('nan'),
            'final_test_acc': float('nan'),
            'final_train_loss': float('nan'),
            'converged': False,
            'elapsed_seconds': elapsed
        }

    final_train_acc = history[-1]['train_acc']
    final_val_acc = history[-1]['val_acc']
    final_test_acc = history[-1]['test_acc']

    return {
        'use_bn': use_bn,
        'optimizer': optimizer_name,
        'lr': lr,
        'seed': seed,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_test_acc': final_test_acc,
        'elapsed_seconds': elapsed,
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(
        description='SimpleMLP BN Ablation - Isolate optimizer vs normalization effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full ablation (publication quality)
    python run_simplemlp_bn_ablation.py --seeds 1,2,3,4,5 --epochs 20

    # Quick test
    python run_simplemlp_bn_ablation.py --quick

    # Ultra-quick (1 seed, 5 epochs)
    python run_simplemlp_bn_ablation.py --ultra-quick
        """
    )

    parser.add_argument('--seeds', type=str, default='1,2,3',
                       help='Comma-separated random seeds (default: 1,2,3)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs (default: 20)')
    parser.add_argument('--results-dir', type=str, default='results/simplemlp_bn_ablation',
                       help='Output directory (default: results/simplemlp_bn_ablation)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 3 seeds, 10 epochs')
    parser.add_argument('--ultra-quick', action='store_true',
                       help='Ultra-quick mode: 1 seed, 5 epochs')

    args = parser.parse_args()

    # Parse seeds
    if args.ultra_quick:
        seeds = [1]
        epochs = 5
    elif args.quick:
        seeds = [1, 2, 3]
        epochs = 10
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
        epochs = args.epochs

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("="*70)
    logging.info("SIMPLEMLP BATCH NORMALIZATION ABLATION")
    logging.info("="*70)
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Device: {device}")
    logging.info(f"Results directory: {results_dir}")

    # Test configurations
    configurations = [
        # Without BN
        {'use_bn': False, 'optimizer': 'SGD', 'lr': 0.1},
        {'use_bn': False, 'optimizer': 'SGD_Momentum', 'lr': 0.1},
        {'use_bn': False, 'optimizer': 'Adam', 'lr': 0.001},
        # With BN
        {'use_bn': True, 'optimizer': 'SGD', 'lr': 0.1},
        {'use_bn': True, 'optimizer': 'SGD_Momentum', 'lr': 0.1},
        {'use_bn': True, 'optimizer': 'Adam', 'lr': 0.001},
    ]

    results = []

    for config in configurations:
        bn_status = "WITH BN" if config['use_bn'] else "WITHOUT BN"
        logging.info(f"\n--- Testing {config['optimizer']} {bn_status} ---")

        for seed in seeds:
            logging.info(f"  Seed {seed}...")
            result = run_ablation(
                use_bn=config['use_bn'],
                optimizer_name=config['optimizer'],
                lr=config['lr'],
                seed=seed,
                epochs=epochs,
                device=device
            )
            results.append(result)

            logging.info(f"    Final Test Acc: {result['final_test_acc']:.4f} "
                        f"({result['elapsed_seconds']:.1f}s)")

    # Save detailed results
    df = pd.DataFrame([{
        'use_bn': r['use_bn'],
        'optimizer': r['optimizer'],
        'lr': r['lr'],
        'seed': r['seed'],
        'final_train_acc': r['final_train_acc'],
        'final_val_acc': r['final_val_acc'],
        'final_test_acc': r['final_test_acc'],
        'elapsed_seconds': r['elapsed_seconds']
    } for r in results])

    csv_path = results_dir / 'simplemlp_bn_ablation_results.csv'
    from src.utils.file_safety import safe_to_csv
    safe_to_csv(df, str(csv_path), index=False)
    logging.info(f"\n✓ Results saved to {csv_path}")

    # Compute summary statistics
    summary = df.groupby(['use_bn', 'optimizer']).agg({
        'final_test_acc': ['mean', 'std'],
        'elapsed_seconds': 'mean'
    }).reset_index()

    summary_path = results_dir / 'simplemlp_bn_ablation_summary.csv'
    safe_to_csv(summary, str(summary_path), index=False)
    logging.info(f"✓ Summary saved to {summary_path}")

    # Print analysis
    logging.info("\n" + "="*70)
    logging.info("ANALYSIS")
    logging.info("="*70)

    for optimizer in ['SGD', 'SGD_Momentum', 'Adam']:
        no_bn_acc = df[(df['use_bn'] == False) & (df['optimizer'] == optimizer)]['final_test_acc'].mean()
        with_bn_acc = df[(df['use_bn'] == True) & (df['optimizer'] == optimizer)]['final_test_acc'].mean()
        improvement = (with_bn_acc - no_bn_acc) * 100

        logging.info(f"\n{optimizer}:")
        logging.info(f"  Without BN: {no_bn_acc:.4f} ({no_bn_acc*100:.2f}%)")
        logging.info(f"  With BN:    {with_bn_acc:.4f} ({with_bn_acc*100:.2f}%)")
        logging.info(f"  Improvement: {improvement:+.2f} percentage points")

        if optimizer == 'SGD' and improvement > 5:
            logging.info(f"  ⚠️ SGD benefits significantly from BN (>{improvement:.1f}pp)")
            logging.info("     This confounds optimizer comparisons!")
        elif optimizer == 'Adam' and abs(improvement) < 1:
            logging.info("  ✓ Adam performance similar with/without BN")
            logging.info("    This validates Adam's adaptive normalization")

    logging.info("\n✓ SimpleMLP BN ablation complete!")
    logging.info(f"✓ Results: {results_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
