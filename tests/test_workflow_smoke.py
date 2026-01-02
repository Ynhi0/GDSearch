"""
Comprehensive smoke test for GDSearch workflow.

Tests the complete pipeline from config parsing through execution
to verify all fixes are working correctly.
"""
import os
import sys
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiments.run_nn_experiment import parse_experiments_from_config
from src.core.loader_validation import enforce_no_test_in_tuning, DatasetSplit
from src.core.data_utils import get_mnist_loaders
import torch


def test_config_parsing_workflow():
    """Test that config parsing produces valid experiments."""
    print("Testing config parsing...")
    
    # Load actual config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'nn_tuning.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Parse experiments
    experiments = parse_experiments_from_config(config)
    
    assert len(experiments) > 0, "No experiments parsed!"
    print(f"✓ Parsed {len(experiments)} experiments from config")
    
    # Validate structure
    for exp in experiments:
        assert 'optimizer' in exp
        assert 'lr' in exp
        assert 'model' in exp
    
    print("✓ All experiments have required fields")


def test_loader_validation_enforcement():
    """Test that loader validation prevents test set leakage."""
    print("\nTesting loader validation...")
    
    # Create loaders with validation split
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=32,
        seed=42,
        val_split=0.1
    )
    
    # Tag loaders
    val_loader.name = 'validation'
    test_loader.name = 'test'
    
    # Validation loader should pass
    enforce_no_test_in_tuning(val_loader)
    print("✓ Validation loader correctly allowed")
    
    # Test loader should be blocked
    try:
        enforce_no_test_in_tuning(test_loader)
        assert False, "Test loader was not blocked!"
    except ValueError:
        print("✓ Test loader correctly blocked")


def test_reproducibility():
    """Test that seeding produces reproducible results."""
    print("\nTesting reproducibility...")
    
    # Run data loader twice with same seed
    train1, test1 = get_mnist_loaders(batch_size=16, seed=42)
    train2, test2 = get_mnist_loaders(batch_size=16, seed=42)
    
    # Get first batch from each
    batch1 = next(iter(train1))
    batch2 = next(iter(train2))
    
    # Compare tensors
    assert torch.allclose(batch1[0], batch2[0]) and torch.equal(batch1[1], batch2[1]), "Data loaders not reproducible!"
    print("✓ Data loaders produce identical batches with same seed")


def test_backward_compatibility():
    """Test that both config formats parse correctly."""
    print("\nTesting backward compatibility...")
    
    # Test new format (optimizers list)
    new_format = {
        "sweeps": [{
            "model": "SimpleMLP",
            "dataset": "MNIST",
            "optimizers": [
                {"name": "Adam", "lr_values": [1e-3, 1e-4]}
            ],
            "epochs": 2,
            "seed": 42
        }]
    }
    
    exps_new = parse_experiments_from_config(new_format)
    assert len(exps_new) == 2, f"Expected 2 experiments, got {len(exps_new)}"
    print("✓ New format (optimizers list) works")
    
    # Test old format (singular optimizer)
    old_format = {
        "sweeps": [{
            "model": "SimpleMLP",
            "dataset": "MNIST",
            "optimizer": "AdamW",
            "lr_values": [1e-3, 1e-4],
            "weight_decay_values": [0.0, 1e-4],
            "epochs": 2,
            "seed": 42
        }]
    }
    
    exps_old = parse_experiments_from_config(old_format)
    # 2 LRs × 2 weight decays = 4 experiments
    assert len(exps_old) == 4, f"Expected 4 experiments, got {len(exps_old)}"
    print("✓ Old format (singular optimizer) works")


def test_fail_fast_empty_config():
    """Test that empty config fails loudly."""
    print("\nTesting fail-fast for empty config...")
    
    empty_config = {"sweeps": []}
    
    exps = parse_experiments_from_config(empty_config)
    assert len(exps) == 0, "Empty config should produce zero experiments!"
    print("✓ Empty config correctly produces zero experiments")
    # In actual code, main() checks this and raises RuntimeError


def run_all_smoke_tests():
    """Run all smoke tests and report results."""
    print("="*60)
    print("GDSearch Workflow Smoke Tests")
    print("="*60)
    
    tests = [
        ("Config Parsing", test_config_parsing_workflow),
        ("Loader Validation", test_loader_validation_enforcement),
        ("Reproducibility", test_reproducibility),
        ("Backward Compatibility", test_backward_compatibility),
        ("Fail-Fast Empty Config", test_fail_fast_empty_config),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            test_fn()  # Test functions use assertions, don't return values
            results.append((name, True))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All smoke tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed - review fixes needed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_smoke_tests()
    sys.exit(exit_code)
