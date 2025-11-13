"""
Data loading utilities for MNIST and CIFAR-10 using torchvision.
Adds optional deterministic seeding for DataLoader workers and transforms.
"""
from typing import Tuple, Optional
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size: int = 128, num_workers: int = 2, seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for MNIST.
    Normalization uses standard MNIST mean/std.
    If seed is provided, DataLoader workers and RNG are seeded for determinism.
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if seed is not None else None,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if seed is not None else None,
        generator=generator,
    )

    return train_loader, test_loader


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2, seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for CIFAR-10.
    Normalization uses CIFAR-10 mean/std.
    If seed is provided, DataLoader workers and RNG are seeded for determinism.
    """
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

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if seed is not None else None,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if seed is not None else None,
        generator=generator,
    )

    return train_loader, test_loader
