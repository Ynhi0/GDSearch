"""
Medical Dataset Utilities for GDSearch

Provides helpers to load real medical datasets (MedMNIST, Kaggle) or fall back to synthetic data.
This module ensures medical experiments can use real data when available while maintaining
reproducibility with synthetic defaults.
"""
# broad catch intentional - module-level allowlist: medical dataset loaders are optional
# and may raise various import/runtime errors; broad catches are used to provide graceful
# degradation for test/CI environments.

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from src.core.medical_dependencies import HAS_MEDMNIST, require_medmnist


class SyntheticMedicalDataset(Dataset):
    """Synthetic medical imaging dataset for segmentation.

    Generates synthetic medical-like images and binary masks for U-Net training.
    Used as fallback when real medical datasets are not available.
    """
    def __init__(self, num_samples: int = 1000, img_size: int = 128, seed: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed
        logging.info(f"Created SyntheticMedicalDataset: {num_samples} samples, size={img_size}x{img_size}, seed={seed}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use per-index Generator for reproducibility (preferred over RandomState)
        rng = np.random.default_rng(self.seed + idx)
        
        # Generate synthetic medical-like images and masks
        # Create base image with noise
        image = rng.normal(0.5, 0.2, (self.img_size, self.img_size)).astype(np.float32)
        image = np.clip(image, 0, 1)

        # Create synthetic anatomical structures (ellipses, circles)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Add 1-3 random structures
        margin = min(20, self.img_size // 4)
        max_radius = min(30, self.img_size // 3)
        min_radius = min(10, self.img_size // 6)
        
        for _ in range(int(rng.integers(1, 4))):
            center_x = int(rng.integers(margin, max(margin+1, self.img_size-margin)))
            center_y = int(rng.integers(margin, max(margin+1, self.img_size-margin)))
            radius_x = int(rng.integers(min_radius, max(min_radius+1, max_radius)))
            radius_y = int(rng.integers(min_radius, max(min_radius+1, max_radius)))

            y, x = np.ogrid[:self.img_size, :self.img_size]
            dist_from_center = ((x - center_x)**2 / radius_x**2) + \
                               ((y - center_y)**2 / radius_y**2)
            structure = (dist_from_center <= 1).astype(np.float32)
            mask = np.maximum(mask, structure)

        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)    # Add channel dimension

        return image, mask


def load_medmnist_dataset(dataset_name: str = 'pathmnist', split: str = 'train',
                          download: bool = True, root: str = './data',
                          strict: bool = False) -> Optional[Dataset]:
    """Load a MedMNIST dataset if the medmnist package is available.

    Args:
        dataset_name: Name of MedMNIST dataset (e.g., 'pathmnist', 'chestmnist', 'organamnist')
        split: 'train', 'val', or 'test'
        download: Whether to download if not present
        root: Root directory for data storage
        strict: If True, raise error when medmnist unavailable; if False, return None

    Returns:
        MedMNIST dataset instance or None if not available

    Raises:
        MedicalDependencyError: If strict=True and medmnist not available
    """
    if not HAS_MEDMNIST:
        if strict:
            require_medmnist(f"MedMNIST dataset loading ('{dataset_name}')")
        else:
            logging.info("medmnist package not installed. Install with: pip install medmnist")
            return None

    try:
        import medmnist  # type: ignore[reportMissingImports]
        from medmnist import INFO  # type: ignore[reportMissingImports]
        import torchvision.transforms as transforms

        if dataset_name not in INFO:
            logging.warning(f"MedMNIST dataset '{dataset_name}' not found. Available: {list(INFO.keys())}")
            return None

        # Dynamically get the dataset class
        DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])

        # Add transforms to ensure tensor output (prevents PIL.Image collate errors)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0,1]
        ])

        # Load the dataset with transform
        dataset = DataClass(split=split, download=download, root=root, transform=transform)
        logging.info(f"Loaded MedMNIST dataset '{dataset_name}' ({split} split): {len(dataset)} samples")
        return dataset

    except Exception as e:
        logging.warning(f"Failed to load MedMNIST dataset '{dataset_name}': {e}")
        return None


from pathlib import Path as _Path

def load_kaggle_medical_dataset(dataset_path: str | _Path = './data/medical', img_size: int = 224, split_seed: int = 42) -> Optional[Tuple[Dataset, Dataset]]:
    """Load a medical dataset downloaded from Kaggle.

    This is a placeholder/template function. Users should customize based on their
    specific Kaggle dataset structure.

    Args:
        dataset_path: Path to the downloaded Kaggle medical dataset

    Returns:
        Tuple of (train_dataset, test_dataset) or None if not available
    """
    dataset_path = _Path(dataset_path)

    if not dataset_path.exists():
        logging.info(f"Kaggle medical dataset not found at {dataset_path}")
        logging.info("To download, run download_datasets.py with Kaggle credentials set")
        return None

    # Best-practice loader: try common Kaggle dataset layouts and fallback to sensible defaults.
    # - If `train/` + `val/` or `train/` + `test/` exist, use ImageFolder on those directories.
    # - If only a top-level folder with class subfolders exists, use ImageFolder and random-split.
    # - Use a deterministic random split (seeded) so experiments are reproducible.
    try:
        import torchvision.transforms as transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import random_split

        # Use the explicit function parameters `img_size` and `split_seed` (defaults set in signature)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'val'
        test_dir = dataset_path / 'test'

        # Case 1: Explicit train/val or train/test directories
        if train_dir.exists():
            train_ds = ImageFolder(str(train_dir), transform=transform)
            if val_dir.exists():
                val_ds = ImageFolder(str(val_dir), transform=transform)
                logging.info(f"Using Kaggle dataset with explicit train/val: {dataset_path}")
                return train_ds, val_ds
            if test_dir.exists():
                test_ds = ImageFolder(str(test_dir), transform=transform)
                logging.info(f"Using Kaggle dataset with explicit train/test: {dataset_path}")
                return train_ds, test_ds

            # No val/test: create a deterministic split from train
            total = len(train_ds)
            if total < 2:
                logging.warning(f"Not enough samples in {train_dir} to split")
                return None
            val_size = max(1, int(0.2 * total))
            split = [total - val_size, val_size]
            generator = torch.Generator().manual_seed(split_seed)
            train_subset, val_subset = random_split(train_ds, split, generator=generator)
            logging.info(f"Split Kaggle train into train/val ({len(train_subset)}/{len(val_subset)})")
            return train_subset, val_subset

        # Case 2: Top-level class subfolders (ImageFolder at root)
        top_dirs = [p for p in dataset_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
        if any((dataset_path / d).is_dir() for d in dataset_path.iterdir()):
            ds = ImageFolder(str(dataset_path), transform=transform)
            total = len(ds)
            if total < 2:
                logging.warning(f"Not enough samples in {dataset_path} to split")
                return None
            val_size = max(1, int(0.2 * total))
            split = [total - val_size, val_size]
            generator = torch.Generator().manual_seed(split_seed)
            train_subset, val_subset = random_split(ds, split, generator=generator)
            logging.info(f"Used top-level ImageFolder and split into train/val ({len(train_subset)}/{len(val_subset)})")
            return train_subset, val_subset

        logging.warning(f"Kaggle dataset layout not recognized at {dataset_path}")
        logging.warning("Please organize as `train/` and `val/` or a top-level class folder structure")
        return None

    except ImportError:
        logging.info("torchvision not installed. Install with: pip install torchvision")
        return None
    except Exception as e:
        logging.warning(f"Failed to load Kaggle dataset at {dataset_path}: {e}")
        return None


def get_medical_datasets(dataset_type: str = 'medmnist',
                        num_train: int = 500,
                        num_test: int = 100,
                        img_size: int = 128,
                        seed: int = 42,
                        **kwargs) -> Tuple[Dataset, Dataset]:
    """Get medical datasets for training and testing.

    Args:
        dataset_type: 'medmnist' (default), 'synthetic', or 'kaggle'
        num_train: Number of training samples (for synthetic fallback)
        num_test: Number of test samples (for synthetic fallback)
        img_size: Image size (for synthetic fallback)
        seed: Random seed
        **kwargs: Additional arguments (medmnist_name='pathmnist', kaggle_path, etc.)

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if dataset_type == 'medmnist':
        medmnist_name = kwargs.get('medmnist_name', 'pathmnist')
        train_ds = load_medmnist_dataset(medmnist_name, split='train')
        test_ds = load_medmnist_dataset(medmnist_name, split='test')

        if train_ds is not None and test_ds is not None:
            logging.info(f"Using MedMNIST dataset: {medmnist_name}")
            return train_ds, test_ds
        else:
            logging.error(f"Failed to load MedMNIST '{medmnist_name}'. Install with: pip install medmnist")
            logging.warning("Falling back to synthetic data. This is NOT recommended for research.")
            dataset_type = 'synthetic'

    elif dataset_type == 'kaggle':
        kaggle_path = kwargs.get('kaggle_path', './data/medical')
        result = load_kaggle_medical_dataset(kaggle_path)

        if result is not None:
            train_ds, test_ds = result
            logging.info(f"Using Kaggle medical dataset from {kaggle_path}")
            return train_ds, test_ds
        else:
            logging.warning(f"Failed to load Kaggle dataset from {kaggle_path}, falling back to synthetic")
            dataset_type = 'synthetic'

    # Fallback: synthetic dataset (NOT RECOMMENDED for research)
    if dataset_type == 'synthetic':
        logging.warning("Using synthetic medical dataset - NOT RECOMMENDED for research")
        logging.warning("Install MedMNIST with: pip install medmnist")
        train_ds = SyntheticMedicalDataset(num_samples=num_train, img_size=img_size, seed=seed)
        test_ds = SyntheticMedicalDataset(num_samples=num_test, img_size=img_size, seed=seed+1000)
        return train_ds, test_ds

    # Fallback if unknown type
    logging.error(f"Unknown dataset_type '{dataset_type}', falling back to synthetic")
    logging.warning("Using synthetic medical dataset - NOT RECOMMENDED for research")
    train_ds = SyntheticMedicalDataset(num_samples=num_train, img_size=img_size, seed=seed)
    test_ds = SyntheticMedicalDataset(num_samples=num_test, img_size=img_size, seed=seed+1000)
    return train_ds, test_ds
