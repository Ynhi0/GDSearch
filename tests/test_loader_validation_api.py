"""
Unit tests for loader validation module.

Tests both the API and the demo/example code to prevent regressions.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# Import the validation functions
try:
    from src.core.loader_validation import (
        validate_loader_for_tuning,
        enforce_no_test_in_tuning,
        create_validated_loaders,
        DatasetSplit
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.core.loader_validation import (
        validate_loader_for_tuning,
        enforce_no_test_in_tuning,
        create_validated_loaders,
        DatasetSplit
    )


def test_validate_loader_for_tuning_accepts_validation():
    """Test that validation loader is accepted."""
    train_data = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))

    val_subset = Subset(train_data, range(100))
    val_loader = DataLoader(val_subset, batch_size=32)
    val_loader.name = 'validation'

    # Should not raise
    validate_loader_for_tuning(val_loader, 'validation', test_dataset=test_data)


def test_validate_loader_for_tuning_rejects_test():
    """Test that test loader is rejected for tuning."""
    train_data = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))

    test_loader = DataLoader(test_data, batch_size=32)
    test_loader.name = 'test'

    # Should raise ValueError
    with pytest.raises(ValueError, match="suggests test data"):
        validate_loader_for_tuning(test_loader, 'validation', test_dataset=test_data)


def test_enforce_no_test_in_tuning_blocks_test():
    """Test that enforce_no_test_in_tuning blocks test loaders."""
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))
    test_loader = DataLoader(test_data, batch_size=32)
    test_loader.name = 'test'

    # Should raise ValueError
    with pytest.raises(ValueError, match="TEST"):
        enforce_no_test_in_tuning(test_loader)


def test_enforce_no_test_in_tuning_allows_validation():
    """Test that enforce_no_test_in_tuning allows validation loaders."""
    train_data = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    val_subset = Subset(train_data, range(100))
    val_loader = DataLoader(val_subset, batch_size=32)
    val_loader.name = 'validation'

    # Should not raise
    enforce_no_test_in_tuning(val_loader)


def test_demo_code_example_one():
    """
    Test the exact demo code from the module's __main__ section.
    This ensures the demo stays in sync with the API.
    """
    # Create dummy datasets (exactly as in demo)
    train_data = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))

    train_loader = DataLoader(train_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # Test 1: Should pass - validation from training data
    val_subset = Subset(train_data, range(100))
    val_loader = DataLoader(val_subset, batch_size=32)
    val_loader.name = 'validation'

    # This is the corrected call (without train_dataset parameter)
    try:
        validate_loader_for_tuning(val_loader, 'validation',
                                   test_dataset=test_data)
        # Success case
        assert True
    except ValueError as e:
        pytest.fail(f"Test 1 should have passed but raised: {e}")


def test_demo_code_example_two():
    """Test the second demo code example."""
    train_data = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))

    test_loader = DataLoader(test_data, batch_size=32)

    # Test 2: Should fail - test loader used for tuning
    test_loader.name = 'test'

    with pytest.raises(ValueError):
        validate_loader_for_tuning(test_loader, 'validation',
                                   test_dataset=test_data)


def test_demo_code_example_three():
    """Test the third demo code example."""
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))
    test_loader = DataLoader(test_data, batch_size=32)
    test_loader.name = 'test'

    # Test 3: Enforce no test in tuning
    with pytest.raises(ValueError):
        enforce_no_test_in_tuning(test_loader)


if __name__ == '__main__':
    """Run tests manually."""
    print("Running loader validation tests...")

    test_validate_loader_for_tuning_accepts_validation()
    print("✓ test_validate_loader_for_tuning_accepts_validation")

    test_validate_loader_for_tuning_rejects_test()
    print("✓ test_validate_loader_for_tuning_rejects_test")

    test_enforce_no_test_in_tuning_blocks_test()
    print("✓ test_enforce_no_test_in_tuning_blocks_test")

    test_enforce_no_test_in_tuning_allows_validation()
    print("✓ test_enforce_no_test_in_tuning_allows_validation")

    test_demo_code_example_one()
    print("✓ test_demo_code_example_one")

    test_demo_code_example_two()
    print("✓ test_demo_code_example_two")

    test_demo_code_example_three()
    print("✓ test_demo_code_example_three")

    print("\n✓ All tests passed!")
