"""
Data Loading Module for GDSearch.

Handles dataset loading, validation splits, and data provenance tracking.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_mnist_loaders(batch_size: int = 128, val_split: Optional[float] = None, 
                     seed: int = 42, num_workers: int = 0) -> Tuple:
    """
    Load MNIST dataset with optional validation split.
    
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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    if val_split is not None and val_split > 0:
        # Split training set into train and validation
        n_train = len(train_dataset)
        n_val = int(n_train * val_split)
        n_train_actual = n_train - n_val
        
        # Reproducible split
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_train, generator=generator).tolist()
        
        train_indices = indices[:n_train_actual]
        val_indices = indices[n_train_actual:]
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader


def get_cifar10_loaders(batch_size: int = 128, val_split: Optional[float] = None,
                       seed: int = 42, num_workers: int = 0) -> Tuple:
    """
    Load CIFAR-10 dataset with optional validation split.
    
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
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                     transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    if val_split is not None and val_split > 0:
        n_train = len(train_dataset)
        n_val = int(n_train * val_split)
        n_train_actual = n_train - n_val
        
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_train, generator=generator).tolist()
        
        train_indices = indices[:n_train_actual]
        val_indices = indices[n_train_actual:]
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    else:
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
