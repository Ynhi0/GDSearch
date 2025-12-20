#!/usr/bin/env python3
"""
GDSearch Kaggle Dataset Download Script
Downloads all datasets used in GDSearch for Kaggle environment testing.

Usage in Kaggle notebook:
    !python download_datasets_kaggle.py

This script is optimized for Kaggle environment (Python 3.10).
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def download_mnist():
    """Download MNIST dataset."""
    print("📥 Downloading MNIST dataset...")
    data_root = Path('/kaggle/working/data')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_dataset = torchvision.datasets.MNIST(
            root=str(data_root), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=str(data_root), train=False, download=True, transform=transform
        )
        print(f"✓ MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
        return True
    except Exception as e:
        print(f"✗ MNIST failed: {e}")
        return False

def download_cifar10():
    """Download CIFAR-10 dataset."""
    print("📥 Downloading CIFAR-10 dataset...")
    data_root = Path('/kaggle/working/data')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=False, download=True, transform=transform_test
        )
        print(f"✓ CIFAR-10: {len(train_dataset)} train, {len(test_dataset)} test samples")
        return True
    except Exception as e:
        print(f"✗ CIFAR-10 failed: {e}")
        return False

def download_fashion_mnist():
    """Download FashionMNIST dataset."""
    print("📥 Downloading FashionMNIST dataset...")
    data_root = Path('/kaggle/working/data')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    try:
        train_dataset = torchvision.datasets.FashionMNIST(
            root=str(data_root), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=str(data_root), train=False, download=True, transform=transform
        )
        print(f"✓ FashionMNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
        return True
    except Exception as e:
        print(f"✗ FashionMNIST failed: {e}")
        return False

def download_imdb():
    """Download IMDB dataset from HuggingFace."""
    print("📥 Downloading IMDB dataset...")

    try:
        from datasets import load_dataset

        # Load full IMDB dataset (works on Kaggle Python 3.10)
        dataset = load_dataset('imdb')
        print(f"✓ IMDB: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
        return True
    except Exception as e:
        print(f"✗ IMDB failed: {e}")
        return False


def download_medmnist(dataset_name: str = 'pathmnist') -> bool:
    """Attempt to ensure medmnist is available on Kaggle (optional real medical dataset)."""
    print("📥 Checking for MedMNIST package on Kaggle...")
    try:
        import medmnist
        from medmnist import INFO
        if dataset_name not in INFO:
            print(f"✗ MedMNIST: dataset '{dataset_name}' not available. Available: {list(INFO.keys())}")
            return False
        print("✓ medmnist is installed on this environment. You can run medical experiments with MedMNIST.")
        return True
    except Exception as e:
        print(f"✗ MedMNIST not available: {e}")
        print("   You can install it in Kaggle notebooks with `pip install medmnist` if needed.")
        return False


def download_medical_real_kaggle(dataset_slug: str = 'paultimothymooney/chest-xray-pneumonia', dest: Path | str = Path('/kaggle/working/data/medical')) -> bool:
    """Try to download a real medical dataset via Kaggle API if possible (Kaggle environment may already have datasets)."""
    print("📥 Attempting optional Kaggle medical dataset download...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("   Kaggle API not available in this environment. Skipping optional medical Kaggle download.")
        return False

    try:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        print(f"   Downloading {dataset_slug} to {dest} (this may take a while)...")
        api.dataset_download_files(dataset_slug, path=str(dest), unzip=True, quiet=False)
        print(f"✓ Medical dataset '{dataset_slug}' downloaded to {dest}")
        return True
    except Exception as e:
        print(f"✗ Kaggle medical download failed: {e}")
        return False


def main():
    """Main download function for Kaggle."""
    print("=" * 60)
    print("GDSearch Kaggle Dataset Download")
    print("=" * 60)

    # Create data directory
    data_dir = Path('/kaggle/working/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Data directory: {data_dir}")

    # Download datasets (including optional medical datasets)
    datasets = [
        ("MNIST", download_mnist),
        ("CIFAR-10", download_cifar10),
        ("FashionMNIST", download_fashion_mnist),
        ("IMDB", download_imdb),
        ("MedMNIST", download_medmnist),
        ("Medical (Kaggle)", download_medical_real_kaggle),
    ]

    results = []
    for name, download_func in datasets:
        print(f"\n{'='*40}")
        print(f"Downloading {name}")
        print('='*40)
        results.append((name, download_func()))

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print('='*60)

    successful = sum(1 for _, success in results if success)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{name:25s}: {status}")

    print(f"\nTotal: {len(results)} datasets (including optional medical datasets)")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

    if successful == len(results):
        print("\n🎉 All datasets ready for GDSearch!")
    else:
        print(f"\n⚠️  {len(results) - successful} dataset(s) failed")

if __name__ == '__main__':
    main()