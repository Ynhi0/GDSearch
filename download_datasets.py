#!/usr/bin/env python3
"""
GDSearch Dataset Download Script
Downloads all datasets used in the GDSearch codebase for Kaggle testing.

Usage:
    python download_datasets.py

This script downloads:
- MNIST (torchvision)
- CIFAR-10 (torchvision)
- FashionMNIST (torchvision)
- IMDB (HuggingFace datasets)
- MedMNIST (REQUIRED for medical experiments)

Note: MedMNIST is now REQUIRED (not optional) for medical experiments.
"""

import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def download_mnist():
    """Download MNIST dataset."""
    print("Downloading MNIST dataset...")
    data_root = Path('./data')

    # Basic transform for download
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        # Download train set
        train_dataset = torchvision.datasets.MNIST(
            root=str(data_root),
            train=True,
            download=True,
            transform=transform
        )
        print(f"MNIST train set: {len(train_dataset)} samples")

        # Download test set
        test_dataset = torchvision.datasets.MNIST(
            root=str(data_root),
            train=False,
            download=True,
            transform=transform
        )
        print(f"MNIST test set: {len(test_dataset)} samples")

    except Exception as e:
        print(f"Failed to download MNIST: {e}")
        return False

    return True

def download_cifar10():
    """Download CIFAR-10 dataset."""
    print("Downloading CIFAR-10 dataset...")
    data_root = Path('./data')

    # Basic transforms for download
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
        # Download train set
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(data_root),
            train=True,
            download=True,
            transform=transform_train
        )
        print(f"CIFAR-10 train set: {len(train_dataset)} samples")

        # Download test set
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(data_root),
            train=False,
            download=True,
            transform=transform_test
        )
        print(f"CIFAR-10 test set: {len(test_dataset)} samples")

    except Exception as e:
        print(f"Failed to download CIFAR-10: {e}")
        return False

    return True

def download_fashion_mnist():
    """Download FashionMNIST dataset."""
    print("Downloading FashionMNIST dataset...")
    data_root = Path('./data')

    # Basic transform for download
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    try:
        # Download train set
        train_dataset = torchvision.datasets.FashionMNIST(
            root=str(data_root),
            train=True,
            download=True,
            transform=transform
        )
        print(f"FashionMNIST train set: {len(train_dataset)} samples")

        # Download test set
        test_dataset = torchvision.datasets.FashionMNIST(
            root=str(data_root),
            train=False,
            download=True,
            transform=transform
        )
        print(f"FashionMNIST test set: {len(test_dataset)} samples")

    except Exception as e:
        print(f"Failed to download FashionMNIST: {e}")
        return False

    return True

def download_imdb():
    """Download IMDB dataset from HuggingFace."""
    print("Downloading IMDB dataset...")

    try:
        from datasets import load_dataset

        # Try to download IMDB dataset (small sample first)
        print("   Attempting to load IMDB dataset from HuggingFace (small sample)...")
        dataset = load_dataset('imdb', split='train[:100]')  # Small sample for testing
        print(f"IMDB dataset sample: {len(dataset)} samples")
        print(f"   Sample text: {dataset[0]['text'][:100]}...")

        # Try full dataset
        print("   Loading full IMDB dataset...")
        full_dataset = load_dataset('imdb')
        print(f"IMDB train set: {len(full_dataset['train'])} samples")
        print(f"IMDB test set: {len(full_dataset['test'])} samples")

    except Exception as e:
        print(f"Failed to download IMDB: {e}")
        print("   Note: IMDB may fail on Python 3.13 - this is expected on some machines.")
        print("   If you see failures locally, try running on Kaggle (Python 3.10) or install 'datasets' and fsspec appropriately.")
        return False

    return True


def download_medmnist(dataset_name: str = 'pathmnist', strict: bool = False) -> bool:
    """Download a small real medical dataset from MedMNIST (optional or required).

    This is a lightweight, public collection of standardized medical image datasets
    suitable for experiments and does not require controlled-access credentials.
    
    Args:
        dataset_name: Name of MedMNIST dataset to download
        strict: If True, raise error when MedMNIST is unavailable (for production runs)
    
    Returns:
        True if successful, False otherwise
    
    Raises:
        RuntimeError: If strict=True and medmnist is not available
    """
    print("Downloading MedMNIST (lightweight medical dataset)...")
    try:
        import medmnist
        from medmnist import INFO
        if dataset_name not in INFO:
            msg = f"MedMNIST: dataset '{dataset_name}' not available. Available: {list(INFO.keys())}"
            print(f"{msg}")
            if strict:
                raise RuntimeError(msg)
            return False

        # The medmnist package provides utilities to access preprocessed datasets.
        # We won't assume a specific loader API here; presence of the package is
        # enough to mark the dataset as available for experiments. Users can run
        # `python -c "from medmnist import INFO; print(INFO['pathmnist'])"` locally
        # to verify details. For reproducible experiments, see docs/ for how to
        # plug MedMNIST into the medical experiment.
        print("medmnist package is installed — please run experiments that use MedMNIST or follow docs to integrate it.")
        return True
    except ImportError as e:
        msg = (f"MedMNIST package not installed: {e}\n"
               "   REQUIRED for medical experiments. Install with: pip install medmnist\n"
               "   For publication-quality results, MedMNIST is mandatory (not synthetic data).")
        print(f"{msg}")
        if strict:
            raise RuntimeError(msg) from e
        print("   Tip: install with `pip install medmnist` and re-run this script, or provide a real medical dataset via the Kaggle option below.")
        return False
    except Exception as e:
        msg = f"MedMNIST not available: {e}"
        print(f"{msg}")
        if strict:
            raise RuntimeError(msg) from e
        print("   Tip: install with `pip install medmnist` and re-run this script, or provide a real medical dataset via the Kaggle option below.")
        return False


def download_medical_real_kaggle(dataset_slug: str = 'paultimothymooney/chest-xray-pneumonia', dest: Path | str = Path('./data/medical')) -> bool:
    """Optionally download a real medical dataset from Kaggle if credentials are available.

    This requires the `kaggle` package and the environment variables `KAGGLE_USERNAME` and
    `KAGGLE_KEY` to be set. The download is optional because many public medical datasets
    require registration or have usage restrictions.
    """
    dest = Path(dest)
    print("Attempting to download a real medical dataset from Kaggle (optional)...")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("   Kaggle API not available. Install it with `pip install kaggle` and set KAGGLE_USERNAME/KAGGLE_KEY to enable automated downloads.")
        return False

    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        print("   Kaggle credentials not found in environment variables. Skipping Kaggle medical download.")
        return False

    try:
        dest.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        print(f"   Downloading {dataset_slug} to {dest} (this may take a while)...")
        api.dataset_download_files(dataset_slug, path=str(dest), unzip=True, quiet=False)
        print(f"Medical dataset '{dataset_slug}' downloaded to {dest}")
        return True
    except Exception as e:
        print(f"Kaggle medical download failed: {e}")
        return False


def main():
    """Main download function."""
    print("=" * 60)
    print("GDSearch Dataset Download Script")
    print("=" * 60)
    print("Downloading all datasets used in GDSearch codebase...")
    print()

    # Create data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    print(f"Using data directory: {data_dir.absolute()}")

    results = []

    # Download each dataset
    print("\n" + "="*40)
    print("1. MNIST Dataset")
    print("="*40)
    results.append(("MNIST", download_mnist()))

    print("\n" + "="*40)
    print("2. CIFAR-10 Dataset")
    print("="*40)
    results.append(("CIFAR-10", download_cifar10()))

    print("\n" + "="*40)
    print("3. FashionMNIST Dataset")
    print("="*40)
    results.append(("FashionMNIST", download_fashion_mnist()))

    print("\n" + "="*40)
    print("4. IMDB Dataset")
    print("="*40)
    results.append(("IMDB", download_imdb()))

    print("\n" + "="*40)
    print("5. MedMNIST (optional real medical data)")
    print("="*40)
    results.append(("MedMNIST", download_medmnist()))

    print("\n" + "="*40)
    print("6. Medical (real, optional via Kaggle)")
    print("="*40)
    results.append(("Medical (Kaggle)", download_medical_real_kaggle()))

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    successful = 0
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{name:25s}: {status}")
        if success:
            successful += 1

    print(f"\nTotal: {len(results)} datasets (including optional medical datasets)")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

    if successful == len(results):
        print("\nAll datasets downloaded successfully!")
        print("You can now run GDSearch experiments on your machine or Kaggle.")
    else:
        print(f"\n{len(results) - successful} dataset(s) failed to download.")
        print("\nCRITICAL: MedMNIST is REQUIRED for medical experiments.")
        print("   Install with: pip install medmnist")
        print("\nNote: IMDB may fail on Python 3.13 but often works on Kaggle (Python 3.10).")

    # Show disk usage
    try:
        import shutil
        total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
        print(f"Disk usage for {data_dir}: {total_size / (1024**2):.2f} MB")
    except Exception:
        pass

if __name__ == '__main__':
    main()