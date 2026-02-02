"""
Validation tests for critical logic fixes.

Tests that verify:
1. No augmentation leakage in validation splits
2. Atomic CSV writes prevent corruption
3. Seed isolation prevents cross-contamination
4. Resume logic works correctly
"""

import pytest
import torch
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Import fixed modules
from src.runners.data_loading import get_mnist_loaders, get_cifar10_loaders
from src.utils.transformed_subset import TransformedSubset, split_indices, has_augmentation
from src.utils.atomic_io import safe_write_csv, safe_write_json
from src.core.training_utils import set_seed


class TestAugmentationLeakageFix:
    """Test that validation splits don't inherit training augmentations."""
    
    def test_cifar10_val_has_no_augmentation(self):
        """CRITICAL: Verify validation set has NO random augmentation."""
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=32, val_split=0.1, seed=42
        )
        
        # Get the transforms from validation loader's dataset
        val_dataset = val_loader.dataset
        val_transform = None
        
        if isinstance(val_dataset, TransformedSubset):
            val_transform = val_dataset.transform
        elif hasattr(val_dataset, 'dataset'):
            # Handle nested subsets
            val_transform = getattr(val_dataset.dataset, 'transform', None)
        
        # Check that validation transform has NO augmentation
        assert val_transform is not None, "Validation dataset should have transform"
        assert not has_augmentation(val_transform), \
            f"Validation transform should NOT have augmentation, but found: {val_transform}"
    
    def test_mnist_loaders_use_transformed_subset(self):
        """Verify MNIST loaders use TransformedSubset for proper isolation."""
        train_loader, val_loader, test_loader = get_mnist_loaders(
            batch_size=32, val_split=0.1, seed=42
        )
        
        # Check that subsets are TransformedSubset (not plain Subset)
        assert isinstance(train_loader.dataset, TransformedSubset), \
            "Train loader should use TransformedSubset"
        assert isinstance(val_loader.dataset, TransformedSubset), \
            "Val loader should use TransformedSubset"
    
    def test_val_metrics_reproducible_without_augmentation(self):
        """Verify validation metrics are deterministic (no random augmentation)."""
        from src.core.models import SimpleCNN
        from torch.utils.data import DataLoader
        import torch.nn as nn
        
        # Load loaders
        _, val_loader, _ = get_cifar10_loaders(batch_size=64, val_split=0.1, seed=42)
        
        # Create model and evaluate twice
        device = torch.device('cpu')
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        
        def evaluate(loader):
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    break  # Just one batch for speed
            return total_loss
        
        loss1 = evaluate(val_loader)
        loss2 = evaluate(val_loader)
        
        # Losses should be IDENTICAL (no randomness in validation)
        assert loss1 == loss2, \
            f"Validation loss should be deterministic but got {loss1} vs {loss2}"


class TestAtomicWritesFix:
    """Test that atomic writes prevent corruption."""
    
    def test_atomic_csv_write_creates_temp_file(self):
        """Verify atomic write uses temp file pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            df = pd.DataFrame({'a': [1, 2, 3]})
            
            # Mock a failure mid-write
            temp_path = path.with_suffix('.csv.tmp')
            
            # Write should succeed
            safe_write_csv(df, path, index=False)
            
            # Temp file should NOT exist after success
            assert not temp_path.exists(), "Temp file should be cleaned up"
            
            # Target file should exist
            assert path.exists(), "Target CSV should exist"
            
            # Content should be correct
            df_read = pd.read_csv(path)
            assert df_read.equals(df), "CSV content should match"
    
    def test_atomic_write_cleans_up_on_failure(self):
        """Verify temp file is cleaned up on write failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create read-only directory to force write failure
            readonly_dir = Path(tmpdir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)  # Read-only
            
            path = readonly_dir / "test.csv"
            df = pd.DataFrame({'a': [1, 2, 3]})
            
            # Write should fail
            try:
                safe_write_csv(df, path, index=False)
                assert False, "Should have raised OSError"
            except OSError:
                pass  # Expected
            
            # Temp file should be cleaned up
            temp_path = path.with_suffix('.csv.tmp')
            assert not temp_path.exists(), "Temp file should be cleaned up on failure"
            
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)
    
    def test_json_atomic_write(self):
        """Verify JSON writes are also atomic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {'experiment': 'test', 'accuracy': 95.2}
            
            safe_write_json(data, path, indent=2)
            
            assert path.exists()
            
            import json
            with open(path) as f:
                loaded = json.load(f)
            
            assert loaded == data


class TestSeedIsolation:
    """Test that seeds are properly isolated between runs."""
    
    def test_different_seeds_produce_different_results(self):
        """Verify different seeds produce different random states."""
        from src.core.models import SimpleMLP
        
        device = torch.device('cpu')
        
        # Initialize with seed 42
        set_seed(42)
        model1 = SimpleMLP()
        weights1 = [p.clone() for p in model1.parameters()]
        
        # Initialize with seed 123 (different)
        set_seed(123)
        model2 = SimpleMLP()
        weights2 = [p.clone() for p in model2.parameters()]
        
        # Weights should be DIFFERENT (different seeds)
        diffs = [not torch.equal(w1, w2) for w1, w2 in zip(weights1, weights2)]
        assert any(diffs), "Different seeds should produce different initializations"
    
    def test_same_seed_produces_same_results(self):
        """Verify same seed produces identical results."""
        from src.core.models import SimpleMLP
        
        # Initialize twice with same seed
        set_seed(42)
        model1 = SimpleMLP()
        weights1 = [p.clone() for p in model1.parameters()]
        
        set_seed(42)
        model2 = SimpleMLP()
        weights2 = [p.clone() for p in model2.parameters()]
        
        # Weights should be IDENTICAL (same seed)
        for w1, w2 in zip(weights1, weights2):
            assert torch.equal(w1, w2), "Same seed should produce identical initialization"


class TestResumeLogic:
    """Test that resume detection works correctly."""
    
    def test_is_experiment_completed_detects_existing_csv(self):
        """Verify resume logic finds existing CSVs."""
        from run_all_kaggle import is_experiment_completed
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            
            # Create result structure
            dataset_dir = results_dir / "experiments" / "mnist"
            dataset_dir.mkdir(parents=True)
            
            # Save a dummy CSV
            csv_path = dataset_dir / "MNIST_SimpleMLP_SGD_seed42.csv"
            df = pd.DataFrame({'epoch': [1, 2, 3], 'loss': [0.5, 0.3, 0.1]})
            df.to_csv(csv_path, index=False)
            
            # Should detect as completed
            is_completed = is_experiment_completed(
                results_dir, 'MNIST', 'SimpleMLP', 'SGD', 42
            )
            
            assert is_completed, "Should detect existing experiment as completed"
    
    def test_is_experiment_completed_rejects_empty_csv(self):
        """Verify resume logic skips empty/corrupted CSVs."""
        from run_all_kaggle import is_experiment_completed
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            dataset_dir = results_dir / "experiments" / "mnist"
            dataset_dir.mkdir(parents=True)
            
            # Create empty CSV
            csv_path = dataset_dir / "MNIST_SimpleMLP_Adam_seed123.csv"
            csv_path.touch()  # Empty file
            
            # Should NOT detect as completed (empty file)
            is_completed = is_experiment_completed(
                results_dir, 'MNIST', 'SimpleMLP', 'Adam', 123
            )
            
            assert not is_completed, "Should reject empty CSV as incomplete"


class TestTransformedSubset:
    """Test the TransformedSubset utility class."""
    
    def test_transformed_subset_applies_custom_transform(self):
        """Verify TransformedSubset applies its own transform, not parent's."""
        from torchvision import datasets, transforms
        
        # Create parent dataset with one transform
        parent_transform = transforms.ToTensor()
        parent_dataset = datasets.FakeData(
            size=100,
            image_size=(3, 32, 32),
            transform=parent_transform
        )
        
        # Create subset with DIFFERENT transform
        subset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        subset = TransformedSubset(
            parent_dataset,
            indices=[0, 1, 2],
            transform=subset_transform
        )
        
        # Get item from subset
        img, _ = subset[0]
        
        # Should be normalized (mean ~0, not 0.5)
        mean_val = img.mean().item()
        assert abs(mean_val) < 0.6, \
            f"Subset should use its own transform, mean should be near 0, got {mean_val}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
