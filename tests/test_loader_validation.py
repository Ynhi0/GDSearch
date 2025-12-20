"""
Tests for loader validation utilities to prevent test set leakage.

These tests ensure the safety mechanisms work correctly to prevent
using test data during hyperparameter tuning, which would invalidate
generalization claims.
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from src.core.loader_validation import (
    enforce_no_test_in_tuning,
    validate_loader_for_tuning,
    DatasetSplit
)


def test_enforce_blocks_test_loader():
    """Test that enforce_no_test_in_tuning blocks loaders tagged as test."""
    # Create dummy dataset
    dummy_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 5, (100,)))
    test_loader = DataLoader(dummy_data, batch_size=32)
    
    # Tag as test
    test_loader.name = 'test'
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="TEST data"):
        enforce_no_test_in_tuning(test_loader)


def test_enforce_allows_validation_loader():
    """Test that enforce_no_test_in_tuning allows validation loaders."""
    dummy_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 5, (100,)))
    val_loader = DataLoader(dummy_data, batch_size=32)
    
    # Tag as validation
    val_loader.name = 'validation'
    
    # Should not raise
    enforce_no_test_in_tuning(val_loader)


def test_validate_detects_test_dataset_identity():
    """Test that validation detects when val_loader contains test dataset."""
    train_data = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))
    
    # Create val_loader that incorrectly uses test_data
    val_loader = DataLoader(test_data, batch_size=32)
    val_loader.name = 'validation'
    
    # Should raise ValueError detecting dataset identity match
    with pytest.raises(ValueError, match="val_loader contains the TEST dataset"):
        validate_loader_for_tuning(
            val_loader,
            expected_split=DatasetSplit.VALIDATION,
            test_dataset=test_data
        )


def test_validate_allows_proper_validation_split():
    """Test that validation passes for proper val split from train."""
    train_data = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))
    
    # Create proper validation subset from training data
    val_subset = Subset(train_data, range(100))
    val_loader = DataLoader(val_subset, batch_size=32)
    val_loader.name = 'validation'
    
    # Should not raise
    validate_loader_for_tuning(
        val_loader,
        expected_split=DatasetSplit.VALIDATION,
        test_dataset=test_data
    )


def test_metadata_tagging():
    """Test that loaders can be tagged with split metadata."""
    dummy_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 5, (100,)))
    loader = DataLoader(dummy_data, batch_size=32)
    
    # Tag with metadata
    loader.name = 'validation'
    setattr(loader, '_split_type', DatasetSplit.VALIDATION)
    
    # Verify metadata
    assert loader.name == 'validation'
    assert getattr(loader, '_split_type') == DatasetSplit.VALIDATION


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
