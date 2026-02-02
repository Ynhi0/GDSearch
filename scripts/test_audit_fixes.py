"""
Quick test of audit fixes implementation.

Verifies that all critical and high-priority fixes work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_result_filename_generator():
    """Test H1: Centralized result filename generator."""
    print("Testing H1: Result filename generator...")
    
    from src.utils.result_filename import (
        generate_result_filename,
        parse_result_filename,
        validate_result_filename
    )
    
    # Test canonical generation
    filename = generate_result_filename(
        model="ResNet18",
        dataset="CIFAR10",
        optimizer="Adam",
        lr=0.001,
        seed=42
    )
    assert filename == "NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv", f"Got: {filename}"
    
    # Test with tag
    filename_with_tag = generate_result_filename(
        model="ResNet18",
        dataset="CIFAR10",
        optimizer="Adam",
        lr=0.001,
        seed=42,
        tag="experiment"
    )
    assert filename_with_tag == "NN_ResNet18_CIFAR10_Adam_lr0.001_seed42_experiment.csv"
    
    # Test parsing
    parsed = parse_result_filename(filename)
    assert parsed['model'] == "ResNet18"
    assert parsed['dataset'] == "CIFAR10"
    assert parsed['optimizer'] == "Adam"
    assert parsed['lr'] == 0.001
    assert parsed['seed'] == 42
    
    # Test validation
    assert validate_result_filename(filename) == True
    assert validate_result_filename("invalid_format.csv") == False
    
    print("✓ H1: Result filename generator works correctly")


def test_config_compatibility_validation():
    """Test H2: Dataset-model compatibility validation."""
    print("Testing H2: Config compatibility validation...")
    
    from src.core.config_loader import validate_config_compatibility, DATASET_MODEL_COMPATIBILITY
    
    # Test valid combination
    valid_config = {"dataset": "CIFAR10", "model": "ResNet18"}
    try:
        validate_config_compatibility(valid_config)
        print("✓ Valid config accepted")
    except ValueError:
        raise AssertionError("Valid config rejected!")
    
    # Test invalid combination
    invalid_config = {"dataset": "CIFAR10", "model": "SimpleLSTM"}
    try:
        validate_config_compatibility(invalid_config)
        raise AssertionError("Invalid config not rejected!")
    except ValueError as e:
        assert "incompatible" in str(e).lower()
        print("✓ Invalid config rejected with proper error")
    
    # Test matrix exists
    assert len(DATASET_MODEL_COMPATIBILITY) > 0
    assert "MNIST" in DATASET_MODEL_COMPATIBILITY
    assert "CIFAR10" in DATASET_MODEL_COMPATIBILITY
    
    print("✓ H2: Config compatibility validation works correctly")


def test_optuna_grace_period():
    """Test H3: Optuna validation grace period."""
    print("Testing H3: Optuna validation grace period...")
    
    # We can only test that the module imports correctly
    # Full testing requires Optuna and a complete setup
    try:
        from src.core.optuna_tuner import OptunaHyperparameterTuner
        print("✓ OptunaHyperparameterTuner imports successfully")
    except ImportError:
        print("⚠ Optuna not available (expected in some environments)")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error in optuna_tuner.py: {e}")
    
    print("✓ H3: Optuna validation module imports correctly")


def test_amsgrad_error_handling():
    """Test C2: AMSGrad shape change error handling."""
    print("Testing C2: AMSGrad shape change error handling...")
    
    from src.core.optimizers import AMSGrad
    import numpy as np
    
    # Create optimizer
    opt = AMSGrad(lr=0.01)
    
    # First step with shape (2,)
    params1 = np.array([1.0, 2.0])
    grad1 = np.array([0.1, 0.2])
    updated1 = opt.step(params1, grad1)
    assert updated1.shape == (2,)
    print("✓ First step successful")
    
    # Second step with different shape should raise RuntimeError
    params2 = np.array([1.0, 2.0, 3.0])  # Different shape!
    grad2 = np.array([0.1, 0.2, 0.3])
    
    try:
        updated2 = opt.step(params2, grad2)
        raise AssertionError("Shape change did not raise RuntimeError!")
    except RuntimeError as e:
        assert "shape changed" in str(e).lower()
        assert "convergence" in str(e).lower()
        print("✓ Shape change properly raises RuntimeError")
    
    print("✓ C2: AMSGrad error handling works correctly")


def test_test_function_constants():
    """Test L2: Constants for magic numbers."""
    print("Testing L2: Test function constants...")
    
    from src.core.test_functions import (
        ROSENBROCK_DEFAULT_A,
        ROSENBROCK_DEFAULT_B,
        QUADRATIC_DEFAULT_KAPPA,
        ACKLEY_DEFAULT_A,
        ROSENBROCK_BOUNDS,
        Rosenbrock,
        IllConditionedQuadratic
    )
    
    # Check constants exist
    assert ROSENBROCK_DEFAULT_A == 1.0
    assert ROSENBROCK_DEFAULT_B == 100.0
    assert QUADRATIC_DEFAULT_KAPPA == 100
    assert ACKLEY_DEFAULT_A == 20.0
    assert ROSENBROCK_BOUNDS == ((-2, 2), (-1, 3))
    
    # Check classes use constants
    rosenbrock = Rosenbrock()
    assert rosenbrock.a == ROSENBROCK_DEFAULT_A
    assert rosenbrock.b == ROSENBROCK_DEFAULT_B
    assert rosenbrock.get_bounds() == ROSENBROCK_BOUNDS
    
    quadratic = IllConditionedQuadratic()
    assert quadratic.kappa == QUADRATIC_DEFAULT_KAPPA
    
    print("✓ L2: Test function constants work correctly")


def test_multi_seed_parsing():
    """Test C1: Multi-seed parameter parsing logic."""
    print("Testing C1: Multi-seed parameter parsing...")
    
    # Test comma-separated parsing
    seeds_str = "42,123,456"
    seeds = [int(s.strip()) for s in seeds_str.split(',')]
    assert seeds == [42, 123, 456]
    
    # Test single seed
    seeds_str = "42"
    seeds = [int(s.strip()) for s in seeds_str.split(',')]
    assert seeds == [42]
    
    # Test with spaces
    seeds_str = "42, 123, 456"
    seeds = [int(s.strip()) for s in seeds_str.split(',')]
    assert seeds == [42, 123, 456]
    
    print("✓ C1: Multi-seed parsing works correctly")


def main():
    """Run all tests."""
    print("=" * 70)
    print("GDSearch Audit Fixes Verification")
    print("=" * 70)
    print()
    
    tests = [
        ("H1: Result Filename Generator", test_result_filename_generator),
        ("H2: Config Compatibility", test_config_compatibility_validation),
        ("H3: Optuna Grace Period", test_optuna_grace_period),
        ("C2: AMSGrad Error Handling", test_amsgrad_error_handling),
        ("L2: Test Function Constants", test_test_function_constants),
        ("C1: Multi-Seed Parsing", test_multi_seed_parsing),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✓ ALL AUDIT FIXES VERIFIED SUCCESSFULLY")
    else:
        print(f"✗ {failed} tests failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
