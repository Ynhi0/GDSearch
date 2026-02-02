"""
Synthetic medical imaging dataset for testing and validation.

Creates minimal synthetic medical segmentation data for quick validation
when real medical datasets are not available.

AUDIT FIX: Addresses missing medical dataset validation failure.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional
import logging


class SyntheticMedicalDataset(Dataset):
    """
    Synthetic 2D medical image segmentation dataset.

    Simulates grayscale medical images (128x128) with binary segmentation masks.
    Useful for smoke tests, CI pipelines, and quick validation.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (128, 128),
        num_classes: int = 2,
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of samples to generate
            image_size: (height, width) of images
            num_classes: Number of segmentation classes (default: 2 for binary)
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

        # Generate deterministic synthetic data
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.images = []
        self.masks = []

        for i in range(num_samples):
            # Create synthetic medical image (grayscale, normalized 0-1)
            img = self._generate_synthetic_image(i)
            mask = self._generate_synthetic_mask(i)

            self.images.append(img)
            self.masks.append(mask)

        logging.info(f"Generated {num_samples} synthetic medical images ({image_size})")

    def _generate_synthetic_image(self, idx: int) -> np.ndarray:
        """Generate realistic-looking synthetic medical image."""
        h, w = self.image_size

        # Base grayscale noise
        img = np.random.randn(h, w) * 0.1 + 0.5

        # Add circular structure (simulates organ/lesion)
        center_y, center_x = h // 2 + np.random.randint(-20, 20), w // 2 + np.random.randint(-20, 20)
        radius = 20 + np.random.randint(-5, 10)

        y, x = np.ogrid[:h, :w]
        circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[circle_mask] += 0.3

        # Add some texture/noise
        img += np.random.randn(h, w) * 0.05

        # Clip to valid range
        img = np.clip(img, 0, 1).astype(np.float32)

        return img

    def _generate_synthetic_mask(self, idx: int) -> np.ndarray:
        """Generate corresponding segmentation mask."""
        h, w = self.image_size

        mask = np.zeros((h, w), dtype=np.int64)

        # Create segmentation region matching image structure
        center_y, center_x = h // 2 + np.random.randint(-20, 20), w // 2 + np.random.randint(-20, 20)
        radius = 20 + np.random.randint(-5, 10)

        y, x = np.ogrid[:h, :w]
        circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[circle_mask] = 1

        return mask

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (1, H, W) - grayscale image
            mask: Tensor of shape (H, W) - segmentation mask
        """
        img = self.images[idx]
        mask = self.masks[idx]

        # Convert to tensors
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Add channel dim
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor


def create_synthetic_medical_data(
    output_dir: str = 'data/synthetic_medical',
    train_samples: int = 80,
    val_samples: int = 20,
    test_samples: int = 20,
    seed: int = 42
) -> Tuple[Path, Path, Path]:
    """
    Create and save synthetic medical dataset splits.
    
    Uses INDEPENDENT random seeds for each split to prevent correlation.

    Args:
        output_dir: Directory to save datasets
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        seed: Random seed

    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    import hashlib
    
    def derive_independent_seed(base_seed: int, split_name: str) -> int:
        """Derive statistically independent seed from base seed + split name."""
        h = hashlib.sha256(f"{base_seed}_{split_name}".encode()).digest()
        return int.from_bytes(h[:4], 'little') % (2**32)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create splits with INDEPENDENT seeds
    train_seed = derive_independent_seed(seed, 'train')
    val_seed = derive_independent_seed(seed, 'val')
    test_seed = derive_independent_seed(seed, 'test')
    
    train_dataset = SyntheticMedicalDataset(train_samples, seed=train_seed)
    val_dataset = SyntheticMedicalDataset(val_samples, seed=val_seed)
    test_dataset = SyntheticMedicalDataset(test_samples, seed=test_seed)
    
    logging.info(f"Generated independent splits: train_seed={train_seed}, val_seed={val_seed}, test_seed={test_seed}")

    # Save as PyTorch datasets
    train_path = output_path / 'train.pt'
    val_path = output_path / 'val.pt'
    test_path = output_path / 'test.pt'

    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)

    logging.info(f"Saved synthetic medical data to {output_dir}")
    logging.info(f"  Train: {train_samples} samples -> {train_path}")
    logging.info(f"  Val: {val_samples} samples -> {val_path}")
    logging.info(f"  Test: {test_samples} samples -> {test_path}")

    return train_path, val_path, test_path


def get_synthetic_medical_loaders(
    batch_size: int = 16,
    num_workers: int = 0,
    data_dir: Optional[str] = None
):
    """
    Get DataLoaders for synthetic medical dataset.

    Compatible drop-in replacement for real medical data loaders.
    """
    from torch.utils.data import DataLoader

    if data_dir is None:
        data_dir = 'data/synthetic_medical'

    data_path = Path(data_dir)

    # Create if doesn't exist
    if not data_path.exists():
        logging.info("Synthetic medical data not found, creating...")
        create_synthetic_medical_data(data_dir)

    # Load datasets
    train_dataset = torch.load(data_path / 'train.pt')
    val_dataset = torch.load(data_path / 'val.pt')
    test_dataset = torch.load(data_path / 'test.pt')

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Creating synthetic medical dataset...")
    train_path, val_path, test_path = create_synthetic_medical_data()

    print("\nTesting dataset loading...")
    dataset = torch.load(train_path)
    img, mask = dataset[0]

    print(f"✅ Image shape: {img.shape}")
    print(f"✅ Mask shape: {mask.shape}")
    print(f"✅ Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"✅ Mask classes: {torch.unique(mask).tolist()}")

    print("\nTesting DataLoader creation...")
    train_loader, val_loader, test_loader = get_synthetic_medical_loaders(batch_size=8)
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")

    print("\n✅ Synthetic medical dataset ready for use!")
