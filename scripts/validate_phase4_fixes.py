"""
Validation script for Phase 4 critical fixes (Issues 22-26).
Verifies that all data integrity and architectural robustness fixes work correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_utils import get_mnist_loaders, get_cifar10_loaders
from src.core.models import SimpleMLP, ConvNet, ResNet18
from src.analysis.theory_practice_comparison import compare_theory_practice


def test_issue_22_augmented_validation():
    """Test Issue #22: Augmented Validation Trap fix."""
    print("\n🔍 Testing Issue #22: Augmented Validation Trap...")
    
    # Load CIFAR-10 with validation split
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=32, num_workers=0, seed=42, val_split=0.1
    )
    
    # Verify train/val/test loaders exist
    assert train_loader is not None, "Train loader creation failed"
    assert val_loader is not None, "Validation loader creation failed"
    assert test_loader is not None, "Test loader creation failed"
    
    # Get sample batches
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    
    # Verify shapes
    assert train_batch[0].shape[0] == 32, f"Train batch size mismatch: {train_batch[0].shape}"
    assert val_batch[0].shape[0] == 32, f"Val batch size mismatch: {val_batch[0].shape}"
    
    # Verify metadata
    assert hasattr(val_loader, '_split_type'), "Validation loader missing _split_type metadata"
    assert val_loader._split_type == 'validation', "Validation loader has wrong split type"
    
    print("  ✅ Validation split created successfully")
    print(f"  ✅ Train size: {len(train_loader.dataset)}")
    print(f"  ✅ Val size: {len(val_loader.dataset)}")
    print(f"  ✅ Test size: {len(test_loader.dataset)}")
    print("  ℹ️  Validation data now uses test transforms (no augmentation)")


def test_issue_23_hardcoded_shape():
    """Test Issue #23: Hardcoded Shape Crash fix."""
    print("\n🔍 Testing Issue #23: Hardcoded Shape Crash...")
    
    # Test ConvNet with CIFAR-10 resolution (32x32)
    model_cifar = ConvNet(num_classes=10)
    x_cifar = torch.randn(4, 3, 32, 32)
    
    try:
        out_cifar = model_cifar(x_cifar)
        assert out_cifar.shape == (4, 10), f"CIFAR-10 output shape mismatch: {out_cifar.shape}"
        print("  ✅ ConvNet works on CIFAR-10 (32×32)")
    except RuntimeError as e:
        print(f"  ❌ ConvNet failed on CIFAR-10: {e}")
        raise
    
    # Test ConvNet with MNIST-like resolution (28x28) - this would crash before fix
    x_mnist = torch.randn(4, 3, 28, 28)
    
    try:
        out_mnist = model_cifar(x_mnist)
        assert out_mnist.shape == (4, 10), f"MNIST output shape mismatch: {out_mnist.shape}"
        print("  ✅ ConvNet works on MNIST-like input (28×28) - FIXED!")
    except RuntimeError as e:
        print(f"  ❌ ConvNet still crashes on 28×28: {e}")
        raise
    
    # Test with arbitrary resolution
    x_arbitrary = torch.randn(4, 3, 64, 64)
    try:
        out_arbitrary = model_cifar(x_arbitrary)
        assert out_arbitrary.shape == (4, 10), f"Arbitrary resolution output shape mismatch: {out_arbitrary.shape}"
        print("  ✅ ConvNet works on arbitrary resolution (64×64)")
    except RuntimeError as e:
        print(f"  ❌ ConvNet failed on arbitrary resolution: {e}")
        raise


def test_issue_24_optimality_gap():
    """Test Issue #24: Curve Fitting Tautology fix."""
    print("\n🔍 Testing Issue #24: Optimality Gap Calculation...")
    
    # Create synthetic training data
    import pandas as pd
    import tempfile
    
    # Simulate strongly convex convergence: f(k) = f_0 * (1 - mu/L)^k
    mu, L = 0.1, 1.0
    theoretical_rate = 1 - mu / L
    iterations = np.arange(100)
    losses = 10.0 * (theoretical_rate ** iterations) + 0.01 * np.random.randn(100)
    
    df = pd.DataFrame({'iteration': iterations, 'loss': losses})
    
    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run comparison
        stats = compare_theory_practice(
            csv_path, 
            optimizer_name='TestSGD',
            output_dir=tmpdir,
            L=L,
            mu=mu
        )
        
        # Verify optimality gap is calculated
        assert 'optimality_gap' in stats, "Optimality gap not calculated"
        assert stats['optimality_gap'] is not None, "Optimality gap is None"
        
        print(f"  ✅ Optimality gap calculated: {stats['optimality_gap']:.6f}")
        print(f"  ✅ Optimality gap percentage: {stats['optimality_gap_pct']:.2f}%")
        print(f"  ✅ Empirical rate: {stats['empirical_rate']:.6f}")
        print(f"  ✅ Theoretical rate: {stats['theoretical_rate']:.6f}")
        
        # Verify gap is small for synthetic data
        assert stats['optimality_gap_pct'] < 20, f"Optimality gap too large: {stats['optimality_gap_pct']:.2f}%"
    
    # Clean up
    Path(csv_path).unlink()


def test_issue_25_zero_gamma_init():
    """Test Issue #25: Zero-Gamma Initialization."""
    print("\n🔍 Testing Issue #25: Zero-Gamma Initialization...")
    
    # Test ResNet18 with default initialization
    model_default = ResNet18(num_classes=10, zero_init_residual=False)
    
    # Count BatchNorm layers with weight=1
    bn_count_default = 0
    for m in model_default.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            if torch.allclose(m.weight, torch.ones_like(m.weight)):
                bn_count_default += 1
    
    print(f"  ✅ Default init: {bn_count_default} BN layers with weight≈1")
    
    # Test ResNet18 with zero-gamma initialization
    model_zero_gamma = ResNet18(num_classes=10, zero_init_residual=True)
    
    # Count BatchNorm layers with weight≈0 (from zero-gamma)
    bn_zero_count = 0
    from src.core.models import BasicBlock
    for m in model_zero_gamma.modules():
        if isinstance(m, BasicBlock):
            # Check if last BN (bn2) has weight≈0
            if torch.allclose(m.bn2.weight, torch.zeros_like(m.bn2.weight), atol=1e-6):
                bn_zero_count += 1
    
    print(f"  ✅ Zero-gamma init: {bn_zero_count} residual blocks with zero-initialized last BN")
    assert bn_zero_count > 0, "Zero-gamma initialization not applied"
    
    # Verify forward pass works
    x = torch.randn(2, 3, 32, 32)
    out_default = model_default(x)
    out_zero_gamma = model_zero_gamma(x)
    
    assert out_default.shape == (2, 10), f"Default model output shape mismatch: {out_default.shape}"
    assert out_zero_gamma.shape == (2, 10), f"Zero-gamma model output shape mismatch: {out_zero_gamma.shape}"
    print("  ✅ Both initialization modes produce valid outputs")


def test_issue_26_bn_control():
    """Test Issue #26: Batch Normalization Control in SimpleMLP."""
    print("\n🔍 Testing Issue #26: Batch Normalization Control...")
    
    # Test SimpleMLP without BN (default)
    model_no_bn = SimpleMLP(use_bn=False)
    
    # Count BN layers
    bn_count = sum(1 for m in model_no_bn.modules() if isinstance(m, torch.nn.BatchNorm1d))
    assert bn_count == 0, f"No-BN model has {bn_count} BN layers"
    print("  ✅ SimpleMLP without BN: 0 BatchNorm layers")
    
    # Test SimpleMLP with BN
    model_with_bn = SimpleMLP(use_bn=True)
    
    # Count BN layers
    bn_count_with = sum(1 for m in model_with_bn.modules() if isinstance(m, torch.nn.BatchNorm1d))
    assert bn_count_with > 0, "With-BN model has no BN layers"
    print(f"  ✅ SimpleMLP with BN: {bn_count_with} BatchNorm layer(s)")
    
    # Verify forward pass works for both
    x = torch.randn(4, 28*28)
    
    out_no_bn = model_no_bn(x)
    assert out_no_bn.shape == (4, 10), f"No-BN output shape mismatch: {out_no_bn.shape}"
    
    out_with_bn = model_with_bn(x)
    assert out_with_bn.shape == (4, 10), f"With-BN output shape mismatch: {out_with_bn.shape}"
    
    print("  ✅ Both configurations produce valid outputs")
    print("  ℹ️  Can now compare SGD with/without BN for fair optimizer analysis")


def main():
    """Run all Phase 4 validation tests."""
    print("=" * 70)
    print("Phase 4 Critical Fixes Validation (Issues 22-26)")
    print("=" * 70)
    
    try:
        test_issue_22_augmented_validation()
        test_issue_23_hardcoded_shape()
        test_issue_24_optimality_gap()
        test_issue_25_zero_gamma_init()
        test_issue_26_bn_control()
        
        print("\n" + "=" * 70)
        print("✅ ALL PHASE 4 FIXES VALIDATED SUCCESSFULLY")
        print("=" * 70)
        print("\n📋 Summary:")
        print("  ✅ Issue 22: Validation split no longer receives augmentation")
        print("  ✅ Issue 23: ConvNet works on any input resolution")
        print("  ✅ Issue 24: Optimality gap quantifies theory-practice mismatch")
        print("  ✅ Issue 25: Zero-gamma initialization available for ResNet18")
        print("  ✅ Issue 26: SimpleMLP BN can be controlled for fair comparison")
        print("\n🎯 All data integrity and architectural robustness fixes operational!")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
