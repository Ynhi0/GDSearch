"""
Medical Dataset Utilities for GDSearch

Provides helpers to load real medical datasets (MedMNIST, Kaggle) or fall back to synthetic data.
This module ensures medical experiments can use real data when available while maintaining
reproducibility with synthetic defaults.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticMedicalDataset(Dataset):
    """Synthetic medical imaging dataset for segmentation.
    
    Generates synthetic medical-like images and binary masks for U-Net training.
    Used as fallback when real medical datasets are not available.
    """
    def __init__(self, num_samples: int = 1000, img_size: int = 128, seed: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        np.random.seed(seed)
        logging.info(f"Created SyntheticMedicalDataset: {num_samples} samples, size={img_size}x{img_size}, seed={seed}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic medical-like images and masks
        # Create base image with noise
        image = np.random.normal(0.5, 0.2, (self.img_size, self.img_size)).astype(np.float32)
        image = np.clip(image, 0, 1)

        # Create synthetic anatomical structures (ellipses, circles)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Add 1-3 random structures
        for _ in range(np.random.randint(1, 4)):
            center_x = np.random.randint(20, self.img_size-20)
            center_y = np.random.randint(20, self.img_size-20)
            radius_x = np.random.randint(10, 30)
            radius_y = np.random.randint(10, 30)

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
                          download: bool = True, root: str = './data') -> Optional[Dataset]:
    """Load a MedMNIST dataset if the medmnist package is available.
    
    Args:
        dataset_name: Name of MedMNIST dataset (e.g., 'pathmnist', 'chestmnist', 'organamnist')
        split: 'train', 'val', or 'test'
        download: Whether to download if not present
        root: Root directory for data storage
        
    Returns:
        MedMNIST dataset instance or None if not available
    """
    try:
        import medmnist
        from medmnist import INFO
        
        if dataset_name not in INFO:
            logging.warning(f"MedMNIST dataset '{dataset_name}' not found. Available: {list(INFO.keys())}")
            return None
        
        # Dynamically get the dataset class
        DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
        
        # Load the dataset
        dataset = DataClass(split=split, download=download, root=root)
        logging.info(f"Loaded MedMNIST dataset '{dataset_name}' ({split} split): {len(dataset)} samples")
        return dataset
        
    except ImportError:
        logging.info("medmnist package not installed. Install with: pip install medmnist")
        return None
    except Exception as e:
        logging.warning(f"Failed to load MedMNIST dataset '{dataset_name}': {e}")
        return None


def load_kaggle_medical_dataset(dataset_path: str = './data/medical') -> Optional[Tuple[Dataset, Dataset]]:
    """Load a medical dataset downloaded from Kaggle.
    
    This is a placeholder/template function. Users should customize based on their
    specific Kaggle dataset structure.
    
    Args:
        dataset_path: Path to the downloaded Kaggle medical dataset
        
    Returns:
        Tuple of (train_dataset, test_dataset) or None if not available
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        logging.info(f"Kaggle medical dataset not found at {dataset_path}")
        logging.info("To download, run download_datasets.py with Kaggle credentials set")
        return None
    
    # TODO: Implement specific loader based on your Kaggle dataset structure
    # Example for chest X-ray pneumonia dataset:
    # train_dir = dataset_path / 'train'
    # test_dir = dataset_path / 'test'
    # Use torchvision.datasets.ImageFolder or custom Dataset class
    
    logging.warning(f"Kaggle dataset loader not yet implemented for {dataset_path}")
    logging.warning("Please customize load_kaggle_medical_dataset() for your specific dataset")
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
