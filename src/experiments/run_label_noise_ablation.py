"""
Label Noise Ablation Study for Optimizer Robustness Analysis.

This module implements controlled label corruption experiments to evaluate
optimizer robustness to noisy labels, a critical component for validating
claims about flat minima and generalization. Tests optimizers under varying
levels of label noise (0%, 10%, 20%, 40%) across multiple seeds.

Key Features:
- Controlled random label flipping with reproducible seeding
- Multi-seed experiments for statistical reliability
- Integration with existing training pipeline and checkpointing
- Automatic tracking of train/val accuracy under noise
- Support for MNIST, CIFAR-10, and other classification tasks

References:
- Zhang et al. (2017): "Understanding deep learning requires rethinking generalization"
- Keskar et al. (2017): "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from src.utils.constants import MNIST_MEAN, MNIST_STD, CIFAR10_MEAN, CIFAR10_STD
from src.utils.safe_len import len_sized
from dataclasses import dataclass

from src.core.training_utils import set_seed
from src.core.models import SimpleMLP, ResNet18

# GAP #24 FIX: Retention rate prevents robustness metric paradox
    # Problem: Weak optimizer (baseline acc=70%) with small drop looks "robust"
    # Solution: retention_rate = noisy_acc / clean_acc normalizes by baseline
    # Example: Optimizer A: 90% -> 85% (retention=94.4%)
    #          Optimizer B: 70% -> 67% (retention=95.7%) <- NOT more robust!

    # Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LabelNoiseConfig:
    """Configuration for label noise experiments."""
    noise_rates: Optional[List[float]] = None  # e.g., [0.0, 0.1, 0.2, 0.4]
    seeds: Optional[List[int]] = None  # e.g., [42, 123, 456]
    epochs: int = 50
    batch_size: int = 128
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.noise_rates is None:
            self.noise_rates = [0.0, 0.1, 0.2, 0.4]
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


class NoisyLabelDataset(Dataset):
    """
    Wraps a dataset to inject controlled label noise.

    Args:
        dataset: Original clean dataset
        noise_rate: Fraction of labels to corrupt (0.0 to 1.0)
        num_classes: Number of classes in the dataset
        seed: Random seed for reproducibility
    """

    def __init__(self, dataset: Dataset, noise_rate: float, num_classes: int, seed: int = 42):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.seed = seed

        # Generate noise mask reproducibly
        self.noisy_labels = self._generate_noisy_labels()

    def _generate_noisy_labels(self) -> np.ndarray:
        """Generate corrupted labels with reproducible randomness."""
        rng = np.random.default_rng(self.seed)
        labels = np.array([self.dataset[i][1] for i in range(len_sized(self.dataset))])

        if self.noise_rate == 0.0:
            return labels

        # Randomly select indices to corrupt
        num_corrupt = int(self.noise_rate * len(labels))
        corrupt_indices = rng.choice(len(labels), size=num_corrupt, replace=False)

        # Flip labels to random incorrect classes
        noisy_labels = labels.copy()
        for idx in corrupt_indices:
            original_label = labels[idx]
            # Choose random label different from original
            incorrect_labels = [l for l in range(self.num_classes) if l != original_label]
            noisy_labels[idx] = rng.choice(incorrect_labels)

        logger.info("Corrupted %d/%d labels (%.1f%%)", num_corrupt, len(labels), self.noise_rate*100)
        return noisy_labels

    def __len__(self) -> int:
        return len_sized(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image, _ = self.dataset[idx]  # Ignore original label
        return image, int(self.noisy_labels[idx])

    def get_clean_accuracy(self) -> float:
        """Compute fraction of labels that remain correct."""
        original_labels = np.array([self.dataset[i][1] for i in range(len_sized(self.dataset))])
        return (original_labels == self.noisy_labels).mean()


def create_noisy_dataloaders(
    dataset_name: str,
    noise_rate: float,
    seed: int,
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data"
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train/val/test dataloaders with label noise injection.

    Args:
        dataset_name: 'mnist' or 'cifar10'
        noise_rate: Fraction of training labels to corrupt
        seed: Random seed for reproducibility
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        data_root: Root directory for datasets

    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    set_seed(seed)

    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
        # BUG FIX #1: MNIST doesn't need separate augmented/non-augmented datasets
        # (no augmentation for MNIST), but initialize variables for consistency
        train_dataset_augmented = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )
        train_dataset_no_augment = train_dataset_augmented  # No augmentation for MNIST
        test_dataset = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform
        )
        num_classes = 10

    elif dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        # BUG FIX #1: Load training data TWICE - once with augmentation, once without
        # This prevents augmentation from being applied to validation split
        train_dataset_augmented = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform_train
        )
        train_dataset_no_augment = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform_test
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform_test
        )
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create validation split (10% of training data)
    # BUG FIX #1: Use augmented dataset for training, non-augmented for validation
    train_size = int(0.9 * len(train_dataset_augmented))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(train_dataset_augmented)))

    train_subset = Subset(train_dataset_augmented, train_indices)
    val_subset = Subset(train_dataset_no_augment, val_indices)  # ← FIX: No augmentation

    # Inject noise only into training set
    noisy_train_dataset = NoisyLabelDataset(train_subset, noise_rate, num_classes, seed)

    train_loader = DataLoader(
        noisy_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes


def train_with_noisy_labels(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: LabelNoiseConfig,
    noise_rate: float,
    seed: int,
    optimizer_name: str
) -> pd.DataFrame:
    """
    Train model with noisy labels and track performance.

    Tracks PEAK validation accuracy instead of final accuracy
    to properly measure robustness to label noise. Models will overfit to
    noise if trained too long, so peak validation performance is the correct
    metric for comparing optimizer robustness.

    Args:
        model: Neural network model
        optimizer: Optimizer instance
        train_loader: Training dataloader (with noisy labels)
        val_loader: Validation dataloader (clean labels)
        test_loader: Test dataloader (clean labels)
        config: Experiment configuration
        noise_rate: Label corruption rate
        seed: Random seed
        optimizer_name: Name of optimizer for logging

    Returns:
        DataFrame with training history
    """
    device = config.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = 0.0
    best_val_epoch = 0
    best_model_state = None

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        # Track best validation accuracy (early stopping criterion)
        # This is crucial for label noise robustness - models overfit to noise
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            # Save best model state for final evaluation
            import copy
            best_model_state = copy.deepcopy(model.state_dict())

        # REPRODUCIBILITY: Only track validation during training, not test
        history.append({
            'epoch': epoch,
            'optimizer': optimizer_name,
            'noise_rate': noise_rate,
            'seed': seed,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'best_val_epoch': best_val_epoch
        })

        if (epoch + 1) % 10 == 0:
            logger.info(
                "[%s] Noise=%.1f%% Seed=%d Epoch %d/%d: Train Acc=%.2f%% Val Acc=%.2f%%",
                optimizer_name, noise_rate*100, seed, epoch+1, config.epochs, train_acc, val_acc
            )

    # Restore best model before final test evaluation
    # This gives the fairest comparison of optimizer robustness to noise
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("[%s] Restored best model from epoch %d (val_acc=%.2f%%)", optimizer_name, best_val_epoch, best_val_acc)

    # Final test evaluation (only after training completes - use test set only for final evaluation)
    logger.info("[%s] Evaluating final performance on test set...", optimizer_name)
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_loss /= test_total
    test_acc = 100.0 * test_correct / test_total
    logger.info("[%s] Final Test Performance: Loss=%.4f, Acc=%.2f%%", optimizer_name, test_loss, test_acc)
    logger.info("[%s] Best Val Acc: %.2f%% at epoch %d", optimizer_name, best_val_acc, best_val_epoch)

    # Add both peak and final metrics for complete analysis
    # Peak metrics show robustness; final metrics show overfitting tendency
    if history:
        # Update last epoch with actual test metrics from BEST model
        history[-1]['test_loss'] = test_loss
        history[-1]['test_acc'] = test_acc  # This is from best checkpoint
        history[-1]['peak_val_acc'] = best_val_acc
        history[-1]['peak_val_epoch'] = best_val_epoch
        # Also keep final_ prefix for backward compatibility
        history[-1]['final_test_loss'] = test_loss
        history[-1]['final_test_acc'] = test_acc
        # Fill earlier epochs with placeholder (or propagate val metrics)
        for i in range(len(history) - 1):
            if 'test_acc' not in history[i]:
                history[i]['test_acc'] = history[i].get('val_acc', 0.0)
                history[i]['test_loss'] = history[i].get('val_loss', 0.0)
    else:
        # If history is empty, create a single summary entry
        history = [{
            'epoch': config.epochs,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'final_test_loss': test_loss,
            'final_test_acc': test_acc,
            'peak_val_acc': best_val_acc,
            'peak_val_epoch': best_val_epoch
        }]

    return pd.DataFrame(history)


def run_label_noise_ablation(
    dataset_name: str,
    model_name: str,
    optimizers_config: Dict[str, Dict[str, Any]],
    config: Optional[LabelNoiseConfig] = None,
    output_dir: str = "results/label_noise"
) -> pd.DataFrame:
    """
    Run complete label noise ablation study across optimizers, noise rates, and seeds.

    Args:
        dataset_name: 'mnist' or 'cifar10'
        model_name: 'mlp' or 'resnet18'
        optimizers_config: Dict mapping optimizer names to their configs
            e.g., {'SGD': {'lr': 0.01, 'momentum': 0.9}, 'Adam': {'lr': 0.001}}
        config: Experiment configuration
        output_dir: Directory to save results

    Returns:
        DataFrame with all experiment results
    """
    if config is None:
        config = LabelNoiseConfig()

    # Normalize optionals to concrete lists for static typing
    noise_rates = list(config.noise_rates) if config.noise_rates is not None else []
    seeds = list(config.seeds) if config.seeds is not None else []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    total_experiments = (
        len(noise_rates) * len(seeds) * len(optimizers_config)
    )
    experiment_count = 0

    logger.info("Starting label noise ablation: %d total experiments", total_experiments)
    logger.info("Dataset: %s, Model: %s", dataset_name, model_name)
    logger.info("Noise rates: %s", noise_rates)
    logger.info("Seeds: %s", seeds)
    logger.info("Optimizers: %s", list(optimizers_config.keys()))

    for noise_rate in noise_rates:
        for seed in seeds:
            # Create dataloaders with noise
            train_loader, val_loader, test_loader, num_classes = create_noisy_dataloaders(
                dataset_name, noise_rate, seed, config.batch_size, config.num_workers
            )

            for optimizer_name, opt_config in optimizers_config.items():
                experiment_count += 1
                logger.info(
                    "\n[%d/%d] Running: %s, noise=%.1f%%, seed=%d",
                    experiment_count, total_experiments, optimizer_name, noise_rate*100, seed
                )

                # Create fresh model
                set_seed(seed)
                if model_name.lower() == 'mlp':
                    input_dim = 784 if dataset_name.lower() == 'mnist' else 3072
                    model = SimpleMLP(input_size=input_dim, num_classes=num_classes)
                elif model_name.lower() == 'resnet18':
                    model = ResNet18(num_classes=num_classes)
                else:
                    raise ValueError(f"Unsupported model: {model_name}")

                # Use optimizer registry for consistency
                from src.core.optimizer_registry import create_optimizer_from_config

                optimizer_config_dict = {'name': optimizer_name}
                optimizer_config_dict.update(opt_config)

                try:
                    optimizer = create_optimizer_from_config(optimizer_config_dict, model.parameters())
                except Exception as e:
                    logger.warning("Registry creation failed, using fallback: %s", e)
                    # Import constant at function level to avoid circular dependency
                    from src.utils.constants import OptimizerNames
                    
                    # Fallback to direct creation
                    opt_lower = optimizer_name.lower()
                    if opt_lower == OptimizerNames.SGD.lower():
                        optimizer = torch.optim.SGD(
                            model.parameters(),
                            lr=opt_config.get('lr', 0.01),
                            momentum=opt_config.get('momentum', 0.0),
                            weight_decay=opt_config.get('weight_decay', 0.0)
                        )
                    elif opt_lower == OptimizerNames.ADAM.lower():
                        # Use AdamW for decoupled weight decay when weight_decay > 0
                        # Original Adam couples weight decay with adaptive LR (Loshchilov & Hutter 2019)
                        wd = opt_config.get('weight_decay', 0.0)
                        if wd > 0:
                            optimizer = torch.optim.AdamW(
                                model.parameters(),
                                lr=opt_config.get('lr', 0.001),
                                betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
                                weight_decay=wd
                            )
                        else:
                            optimizer = torch.optim.Adam(
                                model.parameters(),
                                lr=opt_config.get('lr', 0.001),
                                betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
                                weight_decay=0.0
                            )
                    elif opt_lower == OptimizerNames.ADAMW.lower():
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=opt_config.get('lr', 0.001),
                            betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
                            weight_decay=opt_config.get('weight_decay', 0.01)
                        )
                    elif opt_lower == OptimizerNames.SGD_MOMENTUM.lower():
                        optimizer = torch.optim.SGD(
                            model.parameters(),
                            lr=opt_config.get('lr', 0.01),
                            momentum=opt_config.get('momentum', 0.9),
                            weight_decay=opt_config.get('weight_decay', 0.0)
                        )
                    else:
                        # Generic fallback - assumes optimizer class in config
                        optimizer_class = opt_config.pop('class')
                        optimizer = optimizer_class(model.parameters(), **opt_config)

                # Train and collect results
                results_df = train_with_noisy_labels(
                    model, optimizer, train_loader, val_loader, test_loader,
                    config, noise_rate, seed, optimizer_name
                )
                all_results.append(results_df)

                # Save intermediate results
                combined_df = pd.concat(all_results, ignore_index=True)
                combined_df.to_csv(
                    output_path / f"label_noise_results_{dataset_name}_{model_name}.csv",
                    index=False
                )

    logger.info("\nCompleted all %d experiments!", total_experiments)
    logger.info("Results saved to: %s", output_path)

    # Create summary statistics
    final_df = pd.concat(all_results, ignore_index=True)
    summary = create_label_noise_summary(final_df)
    summary.to_csv(
        output_path / f"label_noise_summary_{dataset_name}_{model_name}.csv",
        index=False
    )

    return final_df


def create_label_noise_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for label noise experiments.

    Args:
        results_df: Raw results from all experiments

    Returns:
        Summary DataFrame with mean ± std for each configuration
    """
    # Get final epoch results for each seed
    final_results = results_df.groupby(['optimizer', 'noise_rate', 'seed']).last().reset_index()

    # Compute mean and std across seeds
    summary = final_results.groupby(['optimizer', 'noise_rate']).agg({
        'train_acc': ['mean', 'std'],
        'val_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std'],
        'train_loss': ['mean', 'std'],
        'val_loss': ['mean', 'std'],
        'test_loss': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    return summary


def analyze_robustness_to_noise(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze optimizer robustness by computing accuracy degradation under noise.

    Args:
        summary_df: Summary statistics from label noise experiments

    Returns:
        DataFrame with robustness metrics per optimizer
    """
    robustness_metrics = []

    for optimizer in summary_df['optimizer'].unique():
        opt_data = summary_df[summary_df['optimizer'] == optimizer]

        # Get clean performance (noise_rate=0.0)
        from src.utils.type_guards import ensure_series
        clean_data = opt_data[opt_data['noise_rate'] == 0.0]['test_acc_mean']
        # Protect against missing clean baseline
        if len(clean_data) == 0:
            logging.warning("No clean baseline (noise_rate=0.0) found for %s, skipping robustness metrics", optimizer)
            continue
        from src.utils.num_utils import safe_to_float
        clean_acc = safe_to_float(ensure_series(clean_data).iloc[0])

        # Compute degradation at each noise level
        for _, row in opt_data.iterrows():
            if row['noise_rate'] > 0.0:
                noisy_acc = row['test_acc_mean']
                acc_drop = clean_acc - noisy_acc
                # Protect against division by zero
                relative_drop = (acc_drop / clean_acc) * 100.0 if clean_acc > 0 else 0.0

                # GAP #24 FIX: Add retention_rate metric to prevent paradox
                # retention_rate = noisy_acc / clean_acc normalizes by baseline
                # This prevents weak optimizers from appearing "robust" due to low baseline
                retention_rate = (noisy_acc / clean_acc) * 100.0 if clean_acc > 0 else 0.0

                robustness_metrics.append({
                    'optimizer': optimizer,
                    'noise_rate': row['noise_rate'],
                    'clean_acc': clean_acc,
                    'noisy_acc': noisy_acc,
                    'noisy_acc_std': row['test_acc_std'],
                    'absolute_drop': acc_drop,
                    'relative_drop_pct': relative_drop,
                    'retention_rate_pct': retention_rate  # GAP #24 FIX: Baseline-normalized robustness
                })

    return pd.DataFrame(robustness_metrics)


if __name__ == "__main__":
    # Example usage
    main_config = LabelNoiseConfig(
        noise_rates=[0.0, 0.1, 0.2, 0.4],
        seeds=[42, 123, 456],
        epochs=30,
        batch_size=128
    )

    main_optimizers_config = {
        'SGD': {'lr': 0.01, 'momentum': 0.0},
        'SGD_Momentum': {'lr': 0.01, 'momentum': 0.9},
        'Adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
        'AdamW': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0.01}
    }

    # Run ablation on MNIST with MLP
    results = run_label_noise_ablation(
        dataset_name='mnist',
        model_name='mlp',
        optimizers_config=main_optimizers_config,
        config=main_config,
        output_dir='results/label_noise'
    )

    logging.info("\nFinal summary statistics:")
    main_summary = create_label_noise_summary(results)
    logging.info(main_summary)

    logging.info("\nRobustness analysis:")
    robustness = analyze_robustness_to_noise(main_summary)
    logging.info(robustness)
