"""
TransformedSubset: Proper subset with independent transform.

Solves the critical data augmentation leakage bug where validation/test
splits inherit training augmentations from the parent dataset.

Usage:
    train_transform = transforms.Compose([RandomCrop(...), ToTensor(), ...])
    eval_transform = transforms.Compose([ToTensor(), ...])  # No augmentation
    
    raw_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    
    train_indices, val_indices = split_indices(len(raw_dataset), 0.1, seed=42)
    
    train_subset = TransformedSubset(raw_dataset, train_indices, train_transform)
    val_subset = TransformedSubset(raw_dataset, val_indices, eval_transform)  # ← Correct!
"""

from torch.utils.data import Dataset
from typing import Callable, List, Optional


class TransformedSubset(Dataset):
    """
    Subset of a dataset with independent transform pipeline.
    
    Unlike torch.utils.data.Subset, this allows applying a different
    transform to the subset than the parent dataset has.
    
    This is critical for train/val splits where:
    - Train needs augmentation (RandomCrop, RandomFlip, etc.)
    - Val/Test must NOT have augmentation (only normalization)
    
    Args:
        dataset: Parent dataset (can have transform=None or any transform)
        indices: List of indices to include in subset
        transform: Transform to apply to this subset (overrides parent's transform)
        target_transform: Optional transform for targets
    """
    
    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform
        
        # Store original transforms to restore them if needed
        self._original_transform = getattr(dataset, 'transform', None)
        self._original_target_transform = getattr(dataset, 'target_transform', None)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.indices) or idx < 0:
            raise IndexError(f"Index {idx} out of range for subset of size {len(self.indices)}")
        
        # Get the actual index in parent dataset
        actual_idx = self.indices[idx]
        
        # Thread-safe data retrieval: Use direct data access or deep copy
        # to avoid mutating shared parent dataset state (critical for multi-worker DataLoader)
        
        if hasattr(self.dataset, 'data') and hasattr(self.dataset, 'targets'):
            # Fast path for datasets with direct data access (MNIST, CIFAR, etc.)
            import torch
            import numpy as np
            
            # Get raw data without triggering parent transforms
            if isinstance(self.dataset.data, torch.Tensor):
                data = self.dataset.data[actual_idx].clone()
            elif isinstance(self.dataset.data, np.ndarray):
                data = self.dataset.data[actual_idx].copy()
            else:
                data = self.dataset.data[actual_idx]
            
            # Get target
            if self.dataset.targets is not None:
                if isinstance(self.dataset.targets, torch.Tensor):
                    target = self.dataset.targets[actual_idx].clone()
                elif isinstance(self.dataset.targets, (list, np.ndarray)):
                    target = self.dataset.targets[actual_idx]
                else:
                    target = self.dataset.targets[actual_idx]
            else:
                target = None
        else:
            # Fallback: Use deep copy to avoid shared state mutation (slower but thread-safe)
            import copy
            import torch
            
            with torch.no_grad():
                # Create temporary copy with no transforms
                temp_dataset = copy.copy(self.dataset)
                temp_dataset.transform = None
                temp_dataset.target_transform = None
                
                sample = temp_dataset[actual_idx]
                if isinstance(sample, tuple) and len(sample) >= 2:
                    data, target = sample[0], sample[1]
                else:
                    data, target = sample, None
        
        # Apply subset's own transforms (thread-safe - each worker has its own transform instance)
        if self.transform is not None:
            data = self.transform(data)
        
        if target is not None and self.target_transform is not None:
            target = self.target_transform(target)
        
        if target is not None:
            return data, target
        else:
            return data


def split_indices(total_size: int, val_fraction: float, seed: int = 42) -> tuple:
    """
    Generate deterministic train/val split indices.
    
    Args:
        total_size: Total number of samples
        val_fraction: Fraction to use for validation (0.0-1.0)
        seed: Random seed for reproducibility
    
    Returns:
        (train_indices, val_indices) as lists
    """
    import torch
    
    n_val = int(total_size * val_fraction)
    n_train = total_size - n_val
    
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    return train_indices, val_indices


# Validation helper to check for augmentation in transforms
def has_augmentation(transform) -> bool:
    """
    Check if a transform pipeline contains data augmentation.
    
    Returns True if transform includes RandomCrop, RandomFlip, ColorJitter, etc.
    Used to detect validation/test sets with incorrect augmentation.
    """
    if transform is None:
        return False
    
    # Check for common augmentation transforms
    augmentation_keywords = [
        'RandomCrop', 'RandomResizedCrop',
        'RandomHorizontalFlip', 'RandomVerticalFlip',
        'RandomRotation', 'RandomAffine',
        'ColorJitter', 'RandomGrayscale',
        'RandomPerspective', 'RandomErasing'
    ]
    
    transform_str = str(transform)
    return any(keyword in transform_str for keyword in augmentation_keywords)
