"""
Data loading utilities for MNIST and CIFAR-10 using torchvision.
Adds optional deterministic seeding for DataLoader workers and transforms.
"""
from typing import Tuple, Optional
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size: int = 128, num_workers: int = 2, seed: Optional[int] = None, val_split: Optional[float] = None) -> Tuple[DataLoader, ...]:
    """
    Create train and test DataLoaders for MNIST.
    Normalization uses standard MNIST mean/std.
    If seed is provided, DataLoader workers and RNG are seeded for determinism.
    
    Args:
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker threads
        seed: Random seed for reproducibility
        val_split: If provided, fraction of training data to use for validation (e.g., 0.1 for 10%)
        
    Returns:
        If val_split is None: (train_loader, test_loader)
        If val_split is provided: (train_loader, val_loader, test_loader)
    """
    # Disable multiprocessing on Windows due to pickle issues
    import platform
    if platform.system() == 'Windows':
        num_workers = 0
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

    worker_seed = seed
    def _worker_init_fn(worker_id: int):
        if worker_seed is None:
            return
        base = int(worker_seed) + worker_id
        np.random.seed(base)
        random.seed(base)
        torch.manual_seed(base)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    # Split train into train/val if requested
    if val_split is not None:
        if not 0.0 < val_split < 1.0:
            raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
        
        total_train = len(full_train_dataset)
        val_size = int(total_train * val_split)
        train_size = total_train - val_size
        
        # Use same generator for reproducible splits
        split_generator = torch.Generator()
        if seed is not None:
            split_generator.manual_seed(int(seed))
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=split_generator
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        
        return train_loader, val_loader, test_loader
    
    else:
        # Original behavior: only train and test
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )

        return train_loader, test_loader


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2, seed: Optional[int] = None, val_split: Optional[float] = None) -> Tuple[DataLoader, ...]:
    """
    Create train and test DataLoaders for CIFAR-10.
    Normalization uses CIFAR-10 mean/std.
    If seed is provided, DataLoader workers and RNG are seeded for determinism.
    
    Args:
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker threads
        seed: Random seed for reproducibility
        val_split: If provided, fraction of training data to use for validation (e.g., 0.1 for 10%)
        
    Returns:
        If val_split is None: (train_loader, test_loader)
        If val_split is provided: (train_loader, val_loader, test_loader)
    """
    # Disable multiprocessing on Windows due to pickle issues
    import platform
    if platform.system() == 'Windows':
        num_workers = 0
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    worker_seed = seed
    def _worker_init_fn(worker_id: int):
        if worker_seed is None:
            return
        base = int(worker_seed) + worker_id
        np.random.seed(base)
        random.seed(base)
        torch.manual_seed(base)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    # Split train into train/val if requested
    if val_split is not None:
        if not 0.0 < val_split < 1.0:
            raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
        
        total_train = len(full_train_dataset)
        val_size = int(total_train * val_split)
        train_size = total_train - val_size
        
        # Use same generator for reproducible splits
        split_generator = torch.Generator()
        if seed is not None:
            split_generator.manual_seed(int(seed))
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=split_generator
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        
        return train_loader, val_loader, test_loader
    
    else:
        # Original behavior: only train and test
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )

        return train_loader, test_loader
