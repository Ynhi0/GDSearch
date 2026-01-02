"""
Data loading utilities for MNIST and CIFAR-10 using torchvision.
Adds optional deterministic seeding for DataLoader workers and transforms.
"""
from typing import Tuple, Optional
import random
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_data_root() -> str:
    """Get data root directory from environment or use default."""
    return os.environ.get('DATA_ROOT', './data')


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

    data_root = get_data_root()
    full_train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)

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
        
        # CRITICAL FIX (Issue #22): Split indices first, then create separate datasets
        # with appropriate transforms (train gets augmentation, val gets test transform)
        # This prevents "Augmented Validation" trap where validation metrics are noisy
        
        # Use torch's random_split on a dummy dataset to get indices
        from torch.utils.data import TensorDataset
        dummy_dataset = TensorDataset(torch.arange(total_train))
        train_idx_subset, val_idx_subset = random_split(
            dummy_dataset,
            [train_size, val_size],
            generator=split_generator
        )
        
        # Create base dataset without transform for index access
        base_train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=None)
        
        # Wrap with transforms AFTER split to prevent augmentation leakage
        class TransformedSubset(torch.utils.data.Dataset):
            """Subset with explicit transform (prevents augmentation leakage to validation)."""
            def __init__(self, base_dataset, indices, transform):
                self.base_dataset = base_dataset
                self.indices = list(indices)
                self.transform = transform
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                img, label = self.base_dataset[real_idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        # Apply TRAINING transform to train split (with augmentation if any)
        train_dataset = TransformedSubset(base_train_dataset, train_idx_subset.indices, transform_train)
        
        # Apply TEST transform to validation split (NO augmentation - clean deterministic data)
        val_dataset = TransformedSubset(base_train_dataset, val_idx_subset.indices, transform_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        # CRITICAL: Add metadata for test-leakage prevention
        train_loader.name = 'train'
        train_loader._split_type = 'train'
        train_loader._dataset_uid = f'mnist_train_{len(train_dataset)}'
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        # CRITICAL: Add metadata for test-leakage prevention
        val_loader.name = 'validation'
        val_loader._split_type = 'validation'
        val_loader._dataset_uid = f'mnist_val_{len(val_dataset)}'
        val_loader._test_dataset_ref = test_dataset
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        # CRITICAL: Add metadata for test-leakage prevention
        test_loader.name = 'test'
        test_loader._split_type = 'test'
        test_loader._dataset_uid = f'mnist_test_{len(test_dataset)}'
        
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
        # CRITICAL: Add metadata for test-leakage prevention
        train_loader.name = 'train'
        train_loader._split_type = 'train'
        train_loader._dataset_uid = f'mnist_train_{len(full_train_dataset)}'
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        # CRITICAL: Add metadata for test-leakage prevention
        test_loader.name = 'test'
        test_loader._split_type = 'test'
        test_loader._dataset_uid = f'mnist_test_{len(test_dataset)}'

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

    data_root = get_data_root()
    full_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

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
        
        # CRITICAL FIX (Issue #22): Split indices first, then create separate datasets
        # with appropriate transforms (train gets augmentation, val gets test transform)
        # This prevents "Augmented Validation" trap where validation metrics are noisy
        # For CIFAR-10, this is CRITICAL because transform_train has RandomCrop and RandomFlip
        
        # Use torch's random_split on a dummy dataset to get indices
        from torch.utils.data import TensorDataset
        dummy_dataset = TensorDataset(torch.arange(total_train))
        train_idx_subset, val_idx_subset = random_split(
            dummy_dataset,
            [train_size, val_size],
            generator=split_generator
        )
        
        # Create base dataset without transform for index access
        base_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)
        
        # Wrap with transforms AFTER split to prevent augmentation leakage
        class TransformedSubset(torch.utils.data.Dataset):
            """Subset with explicit transform (prevents augmentation leakage to validation)."""
            def __init__(self, base_dataset, indices, transform):
                self.base_dataset = base_dataset
                self.indices = list(indices)
                self.transform = transform
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                img, label = self.base_dataset[real_idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        # Apply TRAINING transform to train split (WITH RandomCrop and RandomFlip)
        train_dataset = TransformedSubset(base_train_dataset, train_idx_subset.indices, transform_train)
        
        # Apply TEST transform to validation split (NO augmentation - clean deterministic data)
        # This ensures validation loss is stable and reflects true performance
        val_dataset = TransformedSubset(base_train_dataset, val_idx_subset.indices, transform_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        # CRITICAL: Add metadata for test-leakage prevention
        train_loader.name = 'train'
        train_loader._split_type = 'train'
        train_loader._dataset_uid = f'cifar10_train_{len(train_dataset)}'
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        # CRITICAL: Add metadata for test-leakage prevention
        val_loader.name = 'validation'
        val_loader._split_type = 'validation'
        val_loader._dataset_uid = f'cifar10_val_{len(val_dataset)}'
        val_loader._test_dataset_ref = test_dataset
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        # CRITICAL: Add metadata for test-leakage prevention
        test_loader.name = 'test'
        test_loader._split_type = 'test'
        test_loader._dataset_uid = f'cifar10_test_{len(test_dataset)}'
        
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
        # CRITICAL: Add metadata for test-leakage prevention
        train_loader.name = 'train'
        train_loader._split_type = 'train'
        train_loader._dataset_uid = f'cifar10_train_{len(full_train_dataset)}'
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if seed is not None else None,
            generator=generator,
        )
        # CRITICAL: Add metadata for test-leakage prevention
        test_loader.name = 'test'
        test_loader._split_type = 'test'
        test_loader._dataset_uid = f'cifar10_test_{len(test_dataset)}'

        return train_loader, test_loader


def get_cifar100_loaders(batch_size: int = 128, num_workers: int = 2, seed: Optional[int] = None, val_split: Optional[float] = None):
    """
    Create train and test DataLoaders for CIFAR-100.
    Normalization uses CIFAR-100 mean/std.
    If seed is provided, DataLoader workers and RNG are seeded for determinism.
    
    Returns:
        If val_split is None: (train_loader, test_loader)
        If val_split is provided: (train_loader, val_loader, test_loader)
    """
    data_root = get_data_root()
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    full_train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
    
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
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    if val_split is not None and val_split > 0.0:
        n_train = len(full_train_dataset)
        n_val = int(n_train * val_split)
        n_train_actual = n_train - n_val
        train_dataset, val_dataset = random_split(full_train_dataset, [n_train_actual, n_val], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_init_fn if seed is not None else None, generator=generator)
        # CRITICAL: Add metadata for test-leakage prevention
        train_loader.name = 'train'
        train_loader._split_type = 'train'
        train_loader._dataset_uid = f'cifar100_train_{len(train_dataset)}'
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_init_fn if seed is not None else None, generator=generator)
        # CRITICAL: Add metadata for test-leakage prevention
        val_loader.name = 'validation'
        val_loader._split_type = 'validation'
        val_loader._dataset_uid = f'cifar100_val_{len(val_dataset)}'
        val_loader._test_dataset_ref = test_dataset
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_init_fn if seed is not None else None, generator=generator)
        # CRITICAL: Add metadata for test-leakage prevention
        test_loader.name = 'test'
        test_loader._split_type = 'test'
        test_loader._dataset_uid = f'cifar100_test_{len(test_dataset)}'
        
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_init_fn if seed is not None else None, generator=generator)
        # CRITICAL: Add metadata for test-leakage prevention
        train_loader.name = 'train'
        train_loader._split_type = 'train'
        train_loader._dataset_uid = f'cifar100_train_{len(full_train_dataset)}'
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_init_fn if seed is not None else None, generator=generator)
        # CRITICAL: Add metadata for test-leakage prevention
        test_loader.name = 'test'
        test_loader._split_type = 'test'
        test_loader._dataset_uid = f'cifar100_test_{len(test_dataset)}'
        
        return train_loader, test_loader
