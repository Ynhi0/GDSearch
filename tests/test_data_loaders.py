"""
Unit tests for data loader contract verification.

Tests that get_mnist_loaders and get_cifar10_loaders return the correct 
tuple arity based on val_split parameter.

This is critical for preventing runtime unpacking errors in experiments.
"""

import pytest
import torch
from src.core.data_utils import get_mnist_loaders, get_cifar10_loaders


def test_mnist_loaders_without_val_split():
    """Test MNIST loaders return 2-tuple when val_split=None."""
    result = get_mnist_loaders(batch_size=32, val_split=None)
    
    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
    
    train_loader, test_loader = result
    
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    
    # Verify loaders are not empty
    assert len(train_loader) > 0
    assert len(test_loader) > 0


def test_mnist_loaders_with_val_split():
    """Test MNIST loaders return 3-tuple when val_split provided."""
    result = get_mnist_loaders(batch_size=32, val_split=0.1)
    
    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
    
    train_loader, val_loader, test_loader = result
    
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    
    # Verify loaders are not empty
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    
    # Verify val split roughly correct (within 20% tolerance)
    total_train_val = len(train_loader.dataset) + len(val_loader.dataset)
    val_fraction = len(val_loader.dataset) / total_train_val
    assert 0.08 <= val_fraction <= 0.12, f"Val split {val_fraction:.2f} not close to 0.1"


def test_cifar10_loaders_without_val_split():
    """Test CIFAR-10 loaders return 2-tuple when val_split=None."""
    result = get_cifar10_loaders(batch_size=32, val_split=None)
    
    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
    
    train_loader, test_loader = result
    
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    
    # Verify loaders are not empty
    assert len(train_loader) > 0
    assert len(test_loader) > 0


def test_cifar10_loaders_with_val_split():
    """Test CIFAR-10 loaders return 3-tuple when val_split provided."""
    result = get_cifar10_loaders(batch_size=32, val_split=0.1)
    
    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
    
    train_loader, val_loader, test_loader = result
    
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    
    # Verify loaders are not empty
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    
    # Verify val split roughly correct (within 20% tolerance)
    total_train_val = len(train_loader.dataset) + len(val_loader.dataset)
    val_fraction = len(val_loader.dataset) / total_train_val
    assert 0.08 <= val_fraction <= 0.12, f"Val split {val_fraction:.2f} not close to 0.1"


def test_mnist_loaders_batch_shapes():
    """Test MNIST batch shapes are correct."""
    train_loader, test_loader = get_mnist_loaders(batch_size=32, val_split=None)
    
    # Get first batch
    inputs, targets = next(iter(train_loader))
    
    # MNIST: 1 channel, 28x28 images
    assert inputs.shape[1] == 1, f"Expected 1 channel, got {inputs.shape[1]}"
    assert inputs.shape[2] == 28, f"Expected height 28, got {inputs.shape[2]}"
    assert inputs.shape[3] == 28, f"Expected width 28, got {inputs.shape[3]}"
    assert inputs.shape[0] <= 32, f"Batch size should be <= 32, got {inputs.shape[0]}"
    assert targets.shape[0] == inputs.shape[0]


def test_cifar10_loaders_batch_shapes():
    """Test CIFAR-10 batch shapes are correct."""
    train_loader, test_loader = get_cifar10_loaders(batch_size=32, val_split=None)
    
    # Get first batch
    inputs, targets = next(iter(train_loader))
    
    # CIFAR-10: 3 channels, 32x32 images
    assert inputs.shape[1] == 3, f"Expected 3 channels, got {inputs.shape[1]}"
    assert inputs.shape[2] == 32, f"Expected height 32, got {inputs.shape[2]}"
    assert inputs.shape[3] == 32, f"Expected width 32, got {inputs.shape[3]}"
    assert inputs.shape[0] <= 32, f"Batch size should be <= 32, got {inputs.shape[0]}"
    assert targets.shape[0] == inputs.shape[0]


@pytest.mark.parametrize("val_split", [0.05, 0.1, 0.2, 0.3])
def test_mnist_val_split_fractions(val_split):
    """Test MNIST validation split with various fractions."""
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=32, val_split=val_split)
    
    total_train_val = len(train_loader.dataset) + len(val_loader.dataset)
    actual_val_fraction = len(val_loader.dataset) / total_train_val
    
    # Allow 20% tolerance
    expected_min = val_split * 0.8
    expected_max = val_split * 1.2
    
    assert expected_min <= actual_val_fraction <= expected_max, \
        f"Val split {actual_val_fraction:.3f} not within 20% of {val_split}"


@pytest.mark.parametrize("val_split", [0.05, 0.1, 0.2, 0.3])
def test_cifar10_val_split_fractions(val_split):
    """Test CIFAR-10 validation split with various fractions."""
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=32, val_split=val_split)
    
    total_train_val = len(train_loader.dataset) + len(val_loader.dataset)
    actual_val_fraction = len(val_loader.dataset) / total_train_val
    
    # Allow 20% tolerance
    expected_min = val_split * 0.8
    expected_max = val_split * 1.2
    
    assert expected_min <= actual_val_fraction <= expected_max, \
        f"Val split {actual_val_fraction:.3f} not within 20% of {val_split}"
