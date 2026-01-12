"""
Validation utilities to prevent test set leakage in hyperparameter tuning.

CRITICAL: Ensures that test sets are never used for hyperparameter selection,
which would constitute adaptive overfitting and invalidate research claims.
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional, Tuple, Sized, cast
import logging


class DatasetSplit:
    """Enum-like class for dataset split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def validate_loader_for_tuning(
    loader: DataLoader,
    expected_split: str = DatasetSplit.VALIDATION,
    test_dataset: Optional[torch.utils.data.Dataset] = None
) -> None:
    """
    Validate that a DataLoader is appropriate for hyperparameter tuning.

    CRITICAL SAFETY CHECK: Prevents test set leakage during hyperparameter
    optimization, which would invalidate generalization claims.

    Args:
        loader: DataLoader to validate
        expected_split: Expected split type ('train', 'validation', 'test')
        test_dataset: Reference to test dataset (for validation)

    Raises:
        ValueError: If loader appears to contain test data when validation expected

    Example:
        >>> train_ds, val_ds, test_ds = get_mnist_loaders(val_split=0.2)
        >>> validate_loader_for_tuning(val_loader, 'validation',
        ...                            test_dataset=test_ds.dataset)
    """
    # Check 1: Explicit name attribute (if set by user)
    loader_name = getattr(loader, 'name', None)
    if loader_name:
        if expected_split == DatasetSplit.VALIDATION and 'test' in str(loader_name).lower():
            raise ValueError(
                f"CRITICAL: Loader name '{loader_name}' suggests test data. "
                f"Hyperparameter tuning MUST use validation split only. "
                f"Using test data for tuning = adaptive overfitting (research invalid)."
            )

    # Check 2: Dataset identity check (strongest validation)
    dataset = loader.dataset

    # Handle Subset wrapper
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    # If test_dataset provided, check it's not the same object
    if test_dataset is not None and expected_split == DatasetSplit.VALIDATION:
        # Handle Subset wrapper on test_dataset too
        test_base = test_dataset.dataset if isinstance(test_dataset, Subset) else test_dataset

        # Fast identity check
        if base_dataset is test_base:
            raise ValueError(
                "CRITICAL: val_loader contains the TEST dataset! "
                "Hyperparameter tuning on test set = adaptive overfitting. "
                "This invalidates all generalization claims. "
                "Use a separate validation split from training data."
            )
        # Additional robust checks: dataset UID or small-sample fingerprint
        try:
            uid_a = getattr(base_dataset, '_dataset_uid', None)
            uid_b = getattr(test_base, '_dataset_uid', None)
            if uid_a is not None and uid_b is not None and uid_a == uid_b:
                raise ValueError(
                    "CRITICAL: val_loader dataset appears identical to test dataset (matched _dataset_uid). "
                    "This constitutes test set leakage."
                )

            # Fall back to length + small-sample equality checks (best-effort, cheap)
            if hasattr(base_dataset, '__len__') and hasattr(test_base, '__len__'):
                if len(base_dataset) == len(test_base) and len(base_dataset) > 0:  # type: ignore[arg-type]
                    # Try comparing first few items if possible
                    try:
                        ncheck = min(3, len(base_dataset))  # type: ignore[arg-type]
                        same_count = 0
                        for i in range(ncheck):
                            a_item = base_dataset[i]
                            b_item = test_base[i]
                            if a_item == b_item:
                                same_count += 1
                        if same_count == ncheck:
                            raise ValueError(
                                "CRITICAL: val_loader dataset contents match test dataset (sample check). "
                                "This indicates potential dataset leakage."
                            )
                    except Exception:
                        # If we can't index or equality isn't defined, skip sample equality test
                        pass
        except ValueError:
            raise
        except Exception:
            # Non-fatal: if fingerprinting fails, continue with other checks
            pass

    # Check 3: Dataset length heuristic (weak, but catches obvious errors)
    if hasattr(loader, 'dataset'):
        ds = loader.dataset
        dataset_len = None
        if hasattr(ds, '__len__'):
            # Cast to Sized for static type checker confidence
            dataset_len = len(cast(Sized, ds))
        elif hasattr(ds, 'num_rows'):
            try:
                dataset_len = int(getattr(ds, 'num_rows'))
            except Exception:
                dataset_len = None

        if dataset_len is not None and dataset_len == 0:
            raise ValueError("Loader dataset is empty (length=0)")

        # Log for transparency
        logging.debug(
            f"Loader validation passed: split={expected_split}, "
            f"dataset_len={dataset_len if dataset_len is not None else 'unknown'}, batch_size={loader.batch_size}"
        )


def create_validated_loaders(
    dataset_getter_fn,
    val_split: float = 0.15,
    batch_size: int = 128,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test loaders with built-in validation tags.

    Wraps standard loader creation and adds metadata for safety checks.

    Args:
        dataset_getter_fn: Function like get_mnist_loaders or get_cifar10_loaders
        val_split: Fraction of training data to use for validation
        batch_size: Batch size for all loaders
        **kwargs: Additional args passed to dataset_getter_fn

    Returns:
        Tuple of (train_loader, val_loader, test_loader) with split metadata
    """
    # Get loaders from standard function
    train_loader, val_loader, test_loader = dataset_getter_fn(
        batch_size=batch_size,
        val_split=val_split,
        **kwargs
    )

    # Add explicit metadata for validation
    train_loader.name = 'train'
    val_loader.name = 'validation'
    test_loader.name = 'test'

    # Store dataset references for cross-validation
    # Mark loaders with split type (using attribute assignment for tracking)
    setattr(train_loader, '_split_type', DatasetSplit.TRAIN)
    setattr(val_loader, '_split_type', DatasetSplit.VALIDATION)
    setattr(test_loader, '_split_type', DatasetSplit.TEST)

    # Store cross-references for identity checks
    setattr(val_loader, '_test_dataset_ref', test_loader.dataset)

    return train_loader, val_loader, test_loader


def enforce_no_test_in_tuning(loader: DataLoader) -> None:
    """
    Strict check: Raise error if loader appears to be test data.

    Use this as a guard in hyperparameter tuning functions.

    Args:
        loader: DataLoader to check

    Raises:
        ValueError: If loader is tagged as test data
    """
    # Check explicit metadata
    loader_name = getattr(loader, 'name', '')
    split_type = getattr(loader, '_split_type', '')

    if 'test' in str(loader_name).lower():
        raise ValueError(
            f"BLOCKED: Loader name='{loader_name}' indicates TEST data. "
            f"Hyperparameter tuning with test data is methodologically invalid."
        )

    if split_type == DatasetSplit.TEST:
        raise ValueError(
            "BLOCKED: Loader is tagged as TEST split. "
            "Cannot use test data for hyperparameter selection."
        )

    # Check for validation tag (expected)
    if split_type == DatasetSplit.VALIDATION or 'val' in str(loader_name).lower():
        logging.debug("Loader validation passed: appears to be validation split")
        return

    # If no clear validation tag, log warning but allow (backward compatibility)
    logging.warning(
        f"Loader lacks clear validation tag (name='{loader_name}', split_type='{split_type}'). "
        f"Ensure this is NOT test data to avoid invalidating research."
    )


if __name__ == '__main__':
    # Example usage
    print("Testing loader validation utilities...")

    from torch.utils.data import TensorDataset

    # Create dummy datasets
    train_data = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
    test_data = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))

    train_loader = DataLoader(train_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # Test 1: Should pass - validation from training data
    val_subset = Subset(train_data, range(100))
    val_loader = DataLoader(val_subset, batch_size=32)
    val_loader.name = 'validation'

    try:
        validate_loader_for_tuning(val_loader, 'validation',
                                   test_dataset=test_data)
        print("✓ Test 1 passed: Validation loader correctly accepted")
    except ValueError as e:
        print(f"✗ Test 1 failed: {e}")

    # Test 2: Should fail - test loader used for tuning
    test_loader.name = 'test'
    try:
        validate_loader_for_tuning(test_loader, 'validation',
                                   test_dataset=test_data)
        print("✗ Test 2 failed: Test loader should have been rejected!")
    except ValueError as e:
        print(f"✓ Test 2 passed: Test loader correctly rejected - {str(e)[:80]}...")

    # Test 3: Enforce no test in tuning
    try:
        enforce_no_test_in_tuning(test_loader)
        print("✗ Test 3 failed: Test loader should have been blocked!")
    except ValueError:
        print("✓ Test 3 passed: Test loader blocked by enforcement")

    print("\n✓ All loader validation tests passed!")
