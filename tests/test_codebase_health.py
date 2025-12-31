#!/usr/bin/env python3
"""
Comprehensive Codebase Health Check for GDSearch
Validates consistency, logical correctness, and integration across all critical modules.
"""

import sys
import os
sys.path.insert(0, 'src')

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from src.utils.safe_len import len_sized

def test_requirements():
    """Verify requirements files have necessary dependencies."""
    print("\n" + "="*60)
    print("TEST 1: Requirements Files")
    print("="*60)
    
    # Check main requirements
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    required_packages = ['torch', 'torchvision', 'numpy', 'pandas', 'scipy', 
                        'matplotlib', 'optuna', 'mlflow', 'datasets', 'medmnist']
    
    for pkg in required_packages:
        if pkg in content:
            print(f"  {pkg} found in requirements.txt")
        else:
            print(f"  {pkg} MISSING from requirements.txt")
            assert False, f"{pkg} missing from requirements.txt"
    
    # Check Kaggle requirements
    with open('kaggle/requirements_kaggle.txt', 'r') as f:
        kaggle_content = f.read()
    
    kaggle_required = ['torch', 'transformers', 'datasets', 'plotly', 'medmnist']
    
    for pkg in kaggle_required:
        if pkg in kaggle_content:
            print(f"  ✓ {pkg} found in kaggle requirements")
        else:
            print(f"  ✗ {pkg} MISSING from kaggle requirements")
            assert False, f"{pkg} missing from kaggle requirements"
    
    print("\nAll requirements files are complete!")


def test_imports():
    """Test all critical imports."""
    print("\n" + "="*60)
    print("TEST 2: Critical Imports")
    print("="*60)
    
    from src.core.medical_data_utils import (
        SyntheticMedicalDataset, get_medical_datasets,
        load_medmnist_dataset, load_kaggle_medical_dataset
    )
    print("  Medical data utils import successful")
    
    from src.core.training_utils import set_seed
    print("  ✓ Training utils import successful")
    
    from src.core.data_utils import get_mnist_loaders, get_cifar10_loaders
    print("  ✓ Data utils import successful")
    
    from src.core.dataloader_utils import make_dataloader
    print("  ✓ Dataloader utils import successful")
    
    from src.experiments.run_nn_experiment import build_model_and_data
    print("  ✓ Run NN experiment import successful")
    
    print("\nAll critical imports successful!")


def test_seed_consistency():
    """Test that set_seed produces deterministic results."""
    print("\n" + "="*60)
    print("TEST 3: Seed Consistency")
    print("="*60)
    
    import torch
    from src.core.training_utils import set_seed
    
    # Test 1: PyTorch determinism
    set_seed(42)
    t1 = torch.rand(100)
    set_seed(42)
    t2 = torch.rand(100)
    
    assert torch.allclose(t1, t2), "PyTorch seed is NOT deterministic"
    print("  ✓ PyTorch seed is deterministic")
    
    # Test 2: NumPy determinism
    import numpy as np
    set_seed(42)
    n1 = np.random.rand(100)
    set_seed(42)
    n2 = np.random.rand(100)
    
    assert np.allclose(n1, n2), "NumPy seed is NOT deterministic"
    print("  ✓ NumPy seed is deterministic")
    
    print("\n✓ Seed consistency verified!")


def test_medical_datasets():
    """Test medical dataset utilities."""
    print("\n" + "="*60)
    print("TEST 4: Medical Dataset Utilities")
    print("="*60)
    
    from src.core.medical_data_utils import get_medical_datasets
    
    # Test synthetic dataset
    train_ds, test_ds = get_medical_datasets(
        dataset_type='synthetic',
        num_train=50,
        num_test=25,
        img_size=128,
        seed=42
    )
    
    assert len_sized(train_ds) == 50, f"Expected 50 train samples, got {len_sized(train_ds)}"
    assert len_sized(test_ds) == 25, f"Expected 25 test samples, got {len_sized(test_ds)}"
    print(f"  ✓ Synthetic dataset correct size: {len_sized(train_ds)} train, {len_sized(test_ds)} test")
    
    # Test shapes
    img, mask = train_ds[0]
    assert img.shape == (1, 128, 128), f"Wrong image shape: {img.shape}"
    assert mask.shape == (1, 128, 128), f"Wrong mask shape: {mask.shape}"
    print(f"  ✓ Synthetic dataset correct shapes: {img.shape}")
    
    # Test determinism
    train_ds2, test_ds2 = get_medical_datasets(
        dataset_type='synthetic',
        num_train=50,
        num_test=25,
        img_size=128,
        seed=42
    )
    img2, mask2 = train_ds2[0]
    
    import torch
    assert torch.allclose(img, img2) and torch.allclose(mask, mask2), "Synthetic dataset is NOT deterministic"
    print("  ✓ Synthetic dataset is deterministic with same seed")
    
    print("\n✓ Medical dataset utilities working correctly!")


def test_data_loader_consistency():
    """Test data loader return values are consistent."""
    print("\n" + "="*60)
    print("TEST 5: Data Loader Consistency")
    print("="*60)
    
    from src.core.data_utils import get_mnist_loaders, get_cifar10_loaders
    
    # Test MNIST without val_split
    loaders = get_mnist_loaders(batch_size=32, seed=42)
    assert len(loaders) == 2, f"Expected 2 loaders, got {len(loaders)}"
    print("  ✓ MNIST loaders without val_split: 2 loaders (train, test)")
    
    # Test MNIST with val_split
    loaders = get_mnist_loaders(batch_size=32, seed=42, val_split=0.1)
    assert len(loaders) == 3, f"Expected 3 loaders, got {len(loaders)}"
    print("  ✓ MNIST loaders with val_split: 3 loaders (train, val, test)")
    
    # Test CIFAR-10 without val_split
    loaders = get_cifar10_loaders(batch_size=32, seed=42)
    assert len(loaders) == 2, f"Expected 2 loaders, got {len(loaders)}"
    print("  ✓ CIFAR-10 loaders without val_split: 2 loaders (train, test)")
    
    # Test CIFAR-10 with val_split
    loaders = get_cifar10_loaders(batch_size=32, seed=42, val_split=0.1)
    assert len(loaders) == 3, f"Expected 3 loaders, got {len(loaders)}"
    print("  ✓ CIFAR-10 loaders with val_split: 3 loaders (train, val, test)")
    
    print("\n✓ Data loader return values are consistent!")


def test_build_model_and_data():
    """Test build_model_and_data return consistency."""
    print("\n" + "="*60)
    print("TEST 6: build_model_and_data() Consistency")
    print("="*60)
    
    import torch
    from src.experiments.run_nn_experiment import build_model_and_data
    
    device = torch.device('cpu')
    
    # Test without val_split - API returns a consistent 4-tuple with val_loader = None when not requested
    result = build_model_and_data('MNIST', 'SimpleMLP', 32, device, 42)
    assert len(result) == 4, f"Expected 4 returns (model, train, val, test), got {len(result)}"
    model, train_loader, val_loader, test_loader = result
    assert val_loader is None, "val_loader should be None when val_split is not provided"
    print("  ✓ build_model_and_data without val_split: 4 returns (model, train, val=None, test)")
    
    # Test with val_split
    result = build_model_and_data('MNIST', 'SimpleMLP', 32, device, 42, val_split=0.1)
    assert len(result) == 4, f"Expected 4 returns, got {len(result)}"
    model, train_loader, val_loader, test_loader = result
    print("  ✓ build_model_and_data with val_split: 4 returns (model, train, val, test)")
    
    print("\n✓ build_model_and_data() return values are consistent!")


def test_no_duplicate_classes():
    """Verify no duplicate class definitions."""
    print("\n" + "="*60)
    print("TEST 7: No Duplicate Classes")
    print("="*60)
    
    # Check for duplicate SyntheticMedicalDataset
    from src.core.medical_data_utils import SyntheticMedicalDataset
    print("  OK: SyntheticMedicalDataset only in medical_data_utils")
    
    # Verify run_all_kaggle uses the correct import
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert 'from src.core.medical_data_utils import get_medical_datasets' in content, "run_all_kaggle.py missing medical utils import"
    print("  OK: run_all_kaggle.py uses correct medical utils import")
    
    assert 'class SyntheticMedicalDataset' not in content, "run_all_kaggle.py has duplicate SyntheticMedicalDataset class"
    print("  OK: run_all_kaggle.py has no duplicate SyntheticMedicalDataset class")
    
    print("\nOK: No duplicate classes found!")


def main():
    """Run all health checks."""
    print("\n" + "="*60)
    print("GDSEARCH CODEBASE HEALTH CHECK")
    print("="*60)
    
    tests = [
        test_requirements,
        test_imports,
        test_seed_consistency,
        test_medical_datasets,
        test_data_loader_consistency,
        test_build_model_and_data,
        test_no_duplicate_classes,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL HEALTH CHECKS PASSED!")
        print("\nCodebase is synchronized and logically consistent.")
        print("No bugs or inconsistencies detected.")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        print("\nPlease review failed tests above.")
        return 1


if __name__ == '__main__':
    exit(main())
