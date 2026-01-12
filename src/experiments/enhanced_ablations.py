"""
Enhanced Ablation Studies with Data Efficiency and Model Scaling

Addresses critical gaps identified in Phase 2 review:
1. Data efficiency: Tests performance on 10%, 25%, 50%, 100% of training data
2. Model scaling: Systematic variation of model depth/width
3. Ceteris paribus: Ensures only one variable changes per experiment

This enables stronger generalization claims and cross-domain validation.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd

from src.core.training_utils import set_seed
from src.core.data_utils import get_mnist_loaders, get_cifar10_loaders


class ScalableCNN(nn.Module):
    """CNN with configurable depth and width for model scaling experiments."""

    def __init__(self, input_channels=1, num_classes=10, width_mult=1.0, num_layers=2):
        """
        Args:
            input_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10)
            num_classes: Number of output classes
            width_mult: Width multiplier (1.0 = baseline, 2.0 = double width)
            num_layers: Number of conv layers (2-5 range)
        """
        super().__init__()

        self.width_mult = width_mult
        self.num_layers = num_layers

        # Calculate channel sizes
        base_channels = [32, 64, 128, 256, 512]
        channels = [int(c * width_mult) for c in base_channels[:num_layers]]

        # Build convolutional layers
        layers = []
        in_ch = input_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # Calculate flattened size (depends on input size and pooling)
        # For 28x28 MNIST: after num_layers 2x2 pools -> 28/(2^num_layers)
        # For 32x32 CIFAR: 32/(2^num_layers)
        self.flatten_size = None  # Will be computed on first forward

        self.classifier = None  # Lazy initialization
        self.num_classes = num_classes
        self.last_channel = channels[-1]

    def forward(self, x):
        x = self.features(x)

        # Lazy classifier initialization
        if self.classifier is None:
            self.flatten_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.classifier = nn.Sequential(
                nn.Linear(self.flatten_size, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_classes)
            ).to(x.device)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_data_fraction_subset(
    dataset,
    fraction: float = 1.0,
    seed: int = 42
) -> Subset:
    """
    Create subset of dataset with given fraction.

    Args:
        dataset: PyTorch dataset
        fraction: Fraction of data to use (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Subset of dataset
    """
    if fraction >= 1.0:
        return dataset

    rng = np.random.default_rng(seed)
    n_samples = len(dataset)
    n_subset = int(n_samples * fraction)

    indices = rng.choice(n_samples, size=n_subset, replace=False)
    # Convert to Python list to satisfy typing for Subset indices
    return Subset(dataset, indices.tolist())


def run_data_efficiency_ablation(
    dataset_name: str = 'mnist',
    optimizer_name: str = 'Adam',
    data_fractions: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    epochs: int = 10,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Test optimizer performance across different data fractions.

    CRITICAL: Ensures only data amount varies; all other factors fixed.

    Args:
        dataset_name: 'mnist' or 'cifar10'
        optimizer_name: Optimizer to test
        data_fractions: List of fractions to test
        seeds: Random seeds for statistical validity
        epochs: Training epochs
        device: Device to use

    Returns:
        DataFrame with results
    """
    if data_fractions is None:
        data_fractions = [0.1, 0.25, 0.5, 1.0]
    if seeds is None:
        seeds = [42, 123, 456]

    logging.info("Running data efficiency ablation: %s on %s", optimizer_name, dataset_name)

    results = []
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Get base loaders ONCE - more efficient than recreating for each fraction
    if dataset_name == 'mnist':
        train_base, _val_loader, test_loader = get_mnist_loaders(batch_size=128, val_split=0.1)
        input_channels = 1
        num_classes = 10
    else:  # cifar10
        train_base, _val_loader, test_loader = get_cifar10_loaders(batch_size=128, val_split=0.1)
        input_channels = 3
        num_classes = 10

    # Cache full dataset to avoid reloading - more efficient than creating subset from loader each time
    full_dataset = train_base.dataset

    for fraction in data_fractions:
        for seed in seeds:
            set_seed(seed)

            # Create data subset from cached dataset (efficient)
            subset = create_data_fraction_subset(full_dataset, fraction, seed)
            train_loader = DataLoader(
                subset,
                batch_size=128,
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            # Create model
            model = ScalableCNN(
                input_channels=input_channels,
                num_classes=num_classes,
                width_mult=1.0,
                num_layers=2
            ).to(device_obj)

            # Create optimizer
            # HYPERPARAMETER FAIRNESS: Using published defaults from original papers
            # See docs/HYPERPARAMETER_FAIRNESS_PROTOCOL.md for justification
            # These hyperparameters follow Strategy C (published defaults with citations)
            if optimizer_name == 'SGD':
                # Krizhevsky et al. ImageNet Classification 2012
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            elif optimizer_name == 'Adam':
                # Kingma & Ba Adam paper 2014 (no weight decay)
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
            elif optimizer_name == 'AdamW':
                # Loshchilov & Hutter AdamW paper 2017
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

            # Train
            criterion = nn.CrossEntropyLoss()
            train_losses = []
            diverged = False
            divergence_reason = None

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device_obj), targets.to(device_obj)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # Check for NaN/Inf loss before backward
                    if not torch.isfinite(loss):
                        diverged = True
                        divergence_reason = f"Non-finite loss at epoch {epoch}"
                        logging.warning("Training diverged: %s", divergence_reason)
                        break

                    loss.backward()

                    # Gradient clipping for stability
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Check for exploding gradients
                    if not torch.isfinite(grad_norm):
                        diverged = True
                        divergence_reason = f"Non-finite gradients at epoch {epoch}"
                        logging.warning("Training diverged: %s", divergence_reason)
                        break

                    optimizer.step()
                    epoch_loss += loss.item()

                if diverged:
                    break

                train_losses.append(epoch_loss / len(train_loader))

            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device_obj), targets.to(device_obj)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)

            test_acc = 100.0 * correct / total
            final_train_loss = train_losses[-1] if train_losses else np.nan

            results.append({
                'optimizer': optimizer_name,
                'data_fraction': fraction,
                'n_samples': len(subset),
                'seed': seed,
                'test_accuracy': test_acc,
                'final_train_loss': final_train_loss,
                'dataset': dataset_name,
                'diverged': diverged,
                'divergence_reason': divergence_reason if divergence_reason else 'None'
            })

            logging.info(
                "  Fraction=%.2f, Seed=%d, Samples=%d, Test Acc=%.2f%%",
                fraction, seed, len(subset), test_acc
            )

    return pd.DataFrame(results)


def run_model_scaling_ablation(
    dataset_name: str = 'mnist',
    optimizer_name: str = 'Adam',
    width_mults: Optional[List[float]] = None,
    depth_layers: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    epochs: int = 10,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Test optimizer performance across different model architectures.

    CRITICAL: Ensures only model size varies; optimizer, data, and training fixed.

    Args:
        dataset_name: 'mnist' or 'cifar10'
        optimizer_name: Optimizer to test
        width_mults: Width multipliers to test
        depth_layers: Number of layers to test
        seeds: Random seeds
        epochs: Training epochs
        device: Device to use

    Returns:
        DataFrame with results
    """
    if width_mults is None:
        width_mults = [0.5, 1.0, 2.0]
    if depth_layers is None:
        depth_layers = [2, 3, 4]
    if seeds is None:
        seeds = [42, 123]

    logging.info("Running model scaling ablation: %s on %s", optimizer_name, dataset_name)

    results = []
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Get loaders
    if dataset_name == 'mnist':
        train_loader, _val_loader, test_loader = get_mnist_loaders(batch_size=128, val_split=0.1)
        input_channels = 1
        num_classes = 10
    else:
        train_loader, _val_loader, test_loader = get_cifar10_loaders(batch_size=128, val_split=0.1)
        input_channels = 3
        num_classes = 10

    for width in width_mults:
        for depth in depth_layers:
            for seed in seeds:
                set_seed(seed)

                # Create model with specific architecture
                model = ScalableCNN(
                    input_channels=input_channels,
                    num_classes=num_classes,
                    width_mult=width,
                    num_layers=depth
                ).to(device_obj)

                # Count parameters
                n_params = sum(p.numel() for p in model.parameters())

                # Create optimizer (same hyperparams for fair comparison)
                # HYPERPARAMETER FAIRNESS: Using published defaults from original papers
                # See docs/HYPERPARAMETER_FAIRNESS_PROTOCOL.md Strategy C
                if optimizer_name == 'SGD':
                    # Krizhevsky et al. 2012
                    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
                elif optimizer_name == 'Adam':
                    # Kingma & Ba 2014 (no weight decay)
                    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
                elif optimizer_name == 'AdamW':
                    # Loshchilov & Hutter 2017
                    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                else:
                    raise ValueError(f"Unknown optimizer: {optimizer_name}")

                # Train
                criterion = nn.CrossEntropyLoss()
                diverged = False
                divergence_reason = None

                for epoch in range(epochs):
                    model.train()
                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(device_obj), targets.to(device_obj)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        # Check for NaN/Inf loss
                        if not torch.isfinite(loss):
                            diverged = True
                            divergence_reason = f"Non-finite loss at epoch {epoch}"
                            logging.warning("Training diverged: %s", divergence_reason)
                            break

                        loss.backward()

                        # Gradient clipping for stability
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        # Check for exploding gradients
                        if not torch.isfinite(grad_norm):
                            diverged = True
                            divergence_reason = f"Non-finite gradients at epoch {epoch}"
                            logging.warning("Training diverged: %s", divergence_reason)
                            break

                        optimizer.step()

                    if diverged:
                        break

                # Evaluate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device_obj), targets.to(device_obj)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(targets).sum().item()
                        total += targets.size(0)

                test_acc = 100.0 * correct / total

                results.append({
                    'optimizer': optimizer_name,
                    'width_mult': width,
                    'num_layers': depth,
                    'n_parameters': n_params,
                    'seed': seed,
                    'test_accuracy': test_acc,
                    'dataset': dataset_name,
                    'diverged': diverged,
                    'divergence_reason': divergence_reason if divergence_reason else 'None'
                })

                logging.info(
                    "  Width=%d, Depth=%d, Params=%d, Seed=%d, Test Acc=%.2f%%",
                    width, depth, n_params, seed, test_acc
                )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Enhanced ablation studies')
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--data-fractions', nargs='+', type=float,
                       default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument('--width-mults', nargs='+', type=float,
                       default=[0.5, 1.0, 2.0])
    parser.add_argument('--depths', nargs='+', type=int,
                       default=[2, 3, 4])
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output-dir', default='results/enhanced_ablations')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    # Run data efficiency ablation
    print("\n" + "="*70)
    print("DATA EFFICIENCY ABLATION")
    print("="*70)
    data_results = run_data_efficiency_ablation(
        dataset_name=args.dataset,
        optimizer_name=args.optimizer,
        data_fractions=args.data_fractions,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device
    )
    data_results.to_csv(output_dir / 'data_efficiency_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'data_efficiency_results.csv'}")

    # Run model scaling ablation
    print("\n" + "="*70)
    print("MODEL SCALING ABLATION")
    print("="*70)
    model_results = run_model_scaling_ablation(
        dataset_name=args.dataset,
        optimizer_name=args.optimizer,
        width_mults=args.width_mults,
        depth_layers=args.depths,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device
    )
    model_results.to_csv(output_dir / 'model_scaling_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'model_scaling_results.csv'}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print("\nData Efficiency (Mean ± Std Test Accuracy):")
    data_summary = data_results.groupby('data_fraction')['test_accuracy'].agg(['mean', 'std'])
    print(data_summary)

    print("\nModel Scaling (Mean ± Std Test Accuracy):")
    model_summary = model_results.groupby(['width_mult', 'num_layers'])['test_accuracy'].agg(['mean', 'std'])
    print(model_summary)


if __name__ == '__main__':
    main()
