"""
Data Loading Module for GDSearch.

Handles dataset loading, validation splits, and data provenance tracking.

CRITICAL FIX: Uses TransformedSubset to prevent augmentation leakage into
validation/test splits. Previously, validation sets inherited training
augmentations (RandomCrop, RandomFlip), artificially inflating metrics.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.utils.constants import MNIST_MEAN, MNIST_STD, CIFAR10_MEAN, CIFAR10_STD

# Import fixed subset class that prevents augmentation leakage
from src.utils.transformed_subset import TransformedSubset, split_indices


def _validate_dataset_not_empty(dataset, dataset_name: str):
    """Validate dataset is not empty before creating DataLoader.
    
    Args:
        dataset: Dataset to validate
        dataset_name: Name for error message
        
    Raises:
        ValueError: If dataset is empty
    """
    if len(dataset) == 0:
        raise ValueError(
            f"{dataset_name} is empty. Check data loading and preprocessing. "
            "Cannot create DataLoader for empty dataset."
        )


def get_mnist_loaders(batch_size: int = 128, val_split: Optional[float] = None, 
                     seed: int = 42, num_workers: int = 0) -> Tuple:
    """
    Load MNIST dataset with optional validation split.
    
    FIXED: Validation split no longer inherits training transforms.
    Both train and val use the same transform for MNIST (no augmentation),
    but this pattern is important for CIFAR-10.
    
    Args:
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation (0.0-1.0)
        seed: Random seed for reproducible splits
        num_workers: Number of data loading workers
    
    Returns:
        If val_split is None: (train_loader, test_loader)
        If val_split is provided: (train_loader, val_loader, test_loader)
    """
    from torchvision import datasets, transforms
    
    # MNIST doesn't typically use augmentation, but we use consistent pattern
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    
    # Load raw datasets without transforms initially
    train_dataset_raw = datasets.MNIST('./data', train=True, download=True, transform=None)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    if val_split is not None and val_split > 0:
        # Split training set into train and validation
        n_train = len(train_dataset_raw)
        train_indices, val_indices = split_indices(n_train, val_split, seed)
        
        # Create subsets with explicit transforms (prevents inheritance bugs)
        train_subset = TransformedSubset(train_dataset_raw, train_indices, transform)
        val_subset = TransformedSubset(train_dataset_raw, val_indices, transform)
        
        # Validate datasets are not empty
        _validate_dataset_not_empty(train_subset, "MNIST training subset")
        _validate_dataset_not_empty(val_subset, "MNIST validation subset")
        _validate_dataset_not_empty(test_dataset, "MNIST test dataset")
        
        # Adjust batch size if needed
        effective_batch_size = min(batch_size, len(train_subset))
        if effective_batch_size < batch_size:
            logging.warning(
                f"MNIST: Batch size {batch_size} > training set size {len(train_subset)}. "
                f"Reducing to {effective_batch_size}."
            )
        
        train_loader = DataLoader(train_subset, batch_size=effective_batch_size, 
                                 shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    else:
        # No validation split - apply transform to full training set
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader


def get_cifar10_loaders(batch_size: int = 128, val_split: Optional[float] = None,
                       seed: int = 42, num_workers: int = 0) -> Tuple:
    """
    Load CIFAR-10 dataset with optional validation split.
    
    CRITICAL FIX: Validation split NO LONGER inherits training augmentations.
    - Train: Uses RandomCrop + RandomHorizontalFlip (augmentation)
    - Val/Test: Uses only ToTensor + Normalize (NO augmentation)
    
    Previous bug: Subset(augmented_dataset) made validation inherit
    RandomCrop/Flip, artificially inflating validation accuracy.
    
    Args:
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducible splits
        num_workers: Number of data loading workers
    
    Returns:
        If val_split is None: (train_loader, test_loader)
        If val_split is provided: (train_loader, val_loader, test_loader)
    """
    from torchvision import datasets, transforms
    
    # Training transform: WITH augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Evaluation transform: NO augmentation (only normalization)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Load raw training data WITHOUT transform (apply later per subset)
    train_dataset_raw = datasets.CIFAR10('./data', train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_eval)
    
    if val_split is not None and val_split > 0:
        n_train = len(train_dataset_raw)
        train_indices, val_indices = split_indices(n_train, val_split, seed)
        
        # CRITICAL: Use TransformedSubset to apply DIFFERENT transforms
        # Train gets augmentation, validation gets ONLY normalization
        train_subset = TransformedSubset(train_dataset_raw, train_indices, transform_train)
        val_subset = TransformedSubset(train_dataset_raw, val_indices, transform_eval)  # ← NO augmentation!
        
        # Validate datasets are not empty
        _validate_dataset_not_empty(train_subset, "CIFAR-10 training subset")
        _validate_dataset_not_empty(val_subset, "CIFAR-10 validation subset")
        _validate_dataset_not_empty(test_dataset, "CIFAR-10 test dataset")
        
        # Adjust batch size if needed
        effective_batch_size = min(batch_size, len(train_subset))
        if effective_batch_size < batch_size:
            logging.warning(
                f"CIFAR-10: Batch size {batch_size} > training set size {len(train_subset)}. "
                f"Reducing to {effective_batch_size}."
            )
        
        train_loader = DataLoader(train_subset, batch_size=effective_batch_size,
                                 shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    else:
        # No validation split - apply training transform to full set
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                         transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader


def validate_dataset_split(train_loader: DataLoader, val_loader: Optional[DataLoader],
                          test_loader: DataLoader) -> Dict[str, Any]:
    """
    Validate dataset splits for proper separation and size.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader (can be None)
        test_loader: Test data loader
    
    Returns:
        Dictionary with validation results
    """
    def get_loader_size(loader):
        return len(loader.dataset) if hasattr(loader, 'dataset') else 0
    
    train_size = get_loader_size(train_loader)
    test_size = get_loader_size(test_loader)
    val_size = get_loader_size(val_loader) if val_loader else 0
    
    total = train_size + val_size + test_size
    
    result = {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'total_size': total,
        'train_ratio': train_size / total if total > 0 else 0,
        'val_ratio': val_size / total if total > 0 else 0,
        'test_ratio': test_size / total if total > 0 else 0,
        'has_validation': val_loader is not None
    }
    
    # Validation checks
    if train_size == 0:
        logging.warning("Training set is empty!")
    if test_size == 0:
        logging.warning("Test set is empty!")
    if val_loader and val_size == 0:
        logging.warning("Validation set is requested but empty!")
    
    return result


def log_dataset_provenance(dataset_name: str, split_info: Dict[str, Any],
                          seed: int, experiment_tracker: Any) -> None:
    """
    Log dataset provenance information for reproducibility.
    
    Args:
        dataset_name: Name of the dataset (MNIST, CIFAR-10, etc.)
        split_info: Dictionary with split information
        seed: Random seed used
        experiment_tracker: ExperimentTracker instance
    """
    provenance = {
        'dataset_name': dataset_name,
        'train_size': split_info['train_size'],
        'val_size': split_info['val_size'],
        'test_size': split_info['test_size'],
        'seed': seed,
        'train_ratio': f"{split_info['train_ratio']:.2%}",
        'val_ratio': f"{split_info['val_ratio']:.2%}",
        'test_ratio': f"{split_info['test_ratio']:.2%}",
    }
    
    if experiment_tracker:
        experiment_tracker.log_params(provenance)
    
    logging.info(f"Dataset: {dataset_name}, Train: {split_info['train_size']}, "
                f"Val: {split_info['val_size']}, Test: {split_info['test_size']}, "
                f"Seed: {seed}")
