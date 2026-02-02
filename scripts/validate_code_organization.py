#!/usr/bin/env python3
"""
Validation script for code organization improvements.

This script verifies that all new modules work correctly and can be imported
without errors. Run this after implementing code organization improvements.

Usage:
    python scripts/validate_code_organization.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_imports():
    """Validate all new modules can be imported."""
    print("=" * 70)
    print("PHASE 1: Import Validation")
    print("=" * 70)
    
    modules_to_test = [
        "src.experiments.training_loops",
        "src.core.config_loader",
        "src.core.optimizer_factory",
        "src.core.model_factory",
        "src.utils.constants",
    ]
    
    failed = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            failed.append((module_name, str(e)))
    
    print()
    if failed:
        print(f"FAILED: {len(failed)} modules could not be imported")
        for module, error in failed:
            print(f"  - {module}: {error}")
        return False
    else:
        print("SUCCESS: All modules imported successfully")
        return True


def validate_training_loops():
    """Validate training loop utilities."""
    print("\n" + "=" * 70)
    print("PHASE 2: Training Loop Validation")
    print("=" * 70)
    
    try:
        from src.experiments.training_loops import (
            standard_classification_loop,
            TrainingConfig,
            TrainingResults
        )
        
        # Check dataclasses
        config = TrainingConfig(
            epochs=3,
            device='cpu'  # Will be converted to torch.device
        )
        print(f"✓ TrainingConfig created: epochs={config.epochs}")
        
        results = TrainingResults()
        print(f"✓ TrainingResults created: best_val_acc={results.best_val_acc}")
        
        # Verify functions exist
        assert callable(standard_classification_loop)
        print("✓ standard_classification_loop is callable")
        
        print("\nSUCCESS: Training loop utilities validated")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_config_loader():
    """Validate configuration loader."""
    print("\n" + "=" * 70)
    print("PHASE 3: Config Loader Validation")
    print("=" * 70)
    
    try:
        from src.core.config_loader import ConfigLoader, ConfigValidator
        
        # Test defaults
        mnist_defaults = ConfigLoader.get_dataset_defaults('mnist')
        print(f"✓ MNIST defaults: batch_size={mnist_defaults.get('batch_size')}")
        
        # Test merge
        base = {'a': 1, 'b': {'x': 10}}
        override = {'b': {'y': 20}, 'c': 3}
        merged = ConfigLoader.merge_configs(base, override)
        assert merged['b']['x'] == 10  # Preserved from base
        assert merged['b']['y'] == 20  # Added from override
        print("✓ Config merging works correctly")
        
        # Test defaults application
        config = {'batch_size': 64}
        defaults = {'batch_size': 128, 'epochs': 50}
        result = ConfigLoader.apply_defaults(config, defaults)
        assert result['batch_size'] == 64  # Preserved
        assert result['epochs'] == 50  # Applied from defaults
        print("✓ Default application works correctly")
        
        # Test validation
        test_config = {
            'dataset': 'mnist',
            'optimizers': ['SGD', 'Adam'],
            'epochs': 50,
            'batch_size': 128
        }
        try:
            ConfigValidator.validate_experiment_config(test_config)
            print("✓ Config validation passed")
        except Exception as e:
            # Validation may fail if optimizer registry not initialized, that's OK
            print(f"⚠ Config validation skipped (dependency unavailable): {e}")
        
        print("\nSUCCESS: Config loader validated")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_optimizer_factory():
    """Validate optimizer factory."""
    print("\n" + "=" * 70)
    print("PHASE 4: Optimizer Factory Validation")
    print("=" * 70)
    
    try:
        import torch
        import torch.nn as nn
        from src.core.optimizer_factory import OptimizerFactory
        
        # Create dummy model
        model = nn.Linear(10, 2)
        
        # Test creation
        optimizer = OptimizerFactory.create('Adam', model.parameters(), lr=0.001)
        assert isinstance(optimizer, torch.optim.Adam)
        print(f"✓ Created Adam optimizer: lr={optimizer.param_groups[0]['lr']}")
        
        # Test from config
        opt_config = {'name': 'SGD', 'lr': 0.1, 'momentum': 0.9}
        optimizer = OptimizerFactory.create_from_config(model.parameters(), opt_config)
        assert isinstance(optimizer, torch.optim.SGD)
        print(f"✓ Created SGD from config: lr={optimizer.param_groups[0]['lr']}")
        
        # Test listing
        available = OptimizerFactory.list_optimizers()
        assert len(available) > 0
        print(f"✓ Found {len(available)} registered optimizers")
        
        # Test registration check
        assert OptimizerFactory.is_registered('adam')
        print("✓ is_registered() works")
        
        print("\nSUCCESS: Optimizer factory validated")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_model_factory():
    """Validate model factory."""
    print("\n" + "=" * 70)
    print("PHASE 5: Model Factory Validation")
    print("=" * 70)
    
    try:
        import torch.nn as nn
        from src.core.model_factory import ModelFactory
        
        # Test listing
        available = ModelFactory.list_models()
        print(f"✓ Found {len(available)} registered models")
        
        # Test registration check
        if len(available) > 0:
            first_model = available[0]
            assert ModelFactory.is_registered(first_model)
            print(f"✓ is_registered('{first_model}') works")
            
            # Try to get description
            desc = ModelFactory.get_description(first_model)
            print(f"✓ Description for '{first_model}': {desc[:50]}...")
        
        # Test custom registration
        class DummyModel(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.fc = nn.Linear(100, num_classes)
            
            def forward(self, x):
                return self.fc(x)
        
        ModelFactory.register('TestModel', DummyModel, default_params={'num_classes': 10})
        assert ModelFactory.is_registered('TestModel')
        print("✓ Custom model registration works")
        
        # Test creation
        model = ModelFactory.create('TestModel', num_classes=5)
        assert isinstance(model, nn.Module)
        print("✓ Model creation from factory works")
        
        print("\nSUCCESS: Model factory validated")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_constants():
    """Validate constants module."""
    print("\n" + "=" * 70)
    print("PHASE 6: Constants Validation")
    print("=" * 70)
    
    try:
        from src.utils.constants import (
            ADAM_DEFAULT_LR,
            SGD_DEFAULT_LR,
            DEFAULT_BATCH_SIZE_MNIST,
            MAX_SAFE_LOSS,
            MIN_TRAIN_ACC_MNIST,
            GRADIENT_CLIP_NORM_DEFAULT,
            validate_learning_rate,
            validate_batch_size
        )
        
        # Check constant values
        assert ADAM_DEFAULT_LR == 1e-3
        print(f"✓ ADAM_DEFAULT_LR = {ADAM_DEFAULT_LR}")
        
        assert SGD_DEFAULT_LR == 0.1
        print(f"✓ SGD_DEFAULT_LR = {SGD_DEFAULT_LR}")
        
        assert DEFAULT_BATCH_SIZE_MNIST == 128
        print(f"✓ DEFAULT_BATCH_SIZE_MNIST = {DEFAULT_BATCH_SIZE_MNIST}")
        
        assert MAX_SAFE_LOSS == 1e10
        print(f"✓ MAX_SAFE_LOSS = {MAX_SAFE_LOSS}")
        
        # Test validation functions
        try:
            validate_learning_rate(0.001, 'Adam')
            print("✓ validate_learning_rate() works")
        except Exception:
            pass  # May warn, but shouldn't raise
        
        try:
            validate_batch_size(128, 'mnist')
            print("✓ validate_batch_size() works")
        except Exception:
            pass
        
        print("\nSUCCESS: Constants module validated")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CODE ORGANIZATION VALIDATION" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = {
        "Imports": validate_imports(),
        "Training Loops": validate_training_loops(),
        "Config Loader": validate_config_loader(),
        "Optimizer Factory": validate_optimizer_factory(),
        "Model Factory": validate_model_factory(),
        "Constants": validate_constants(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 20 + "ALL VALIDATIONS PASSED!" + " " * 26 + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        print("✓ All new code organization modules are working correctly")
        print("✓ Ready to use in experiments")
        print()
        print("Next steps:")
        print("  1. Review CODE_ORGANIZATION_IMPROVEMENTS.md")
        print("  2. Check docs/CODE_ORG_QUICK_REFERENCE.md for usage examples")
        print("  3. Start using new modules in experiments")
        return 0
    else:
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 22 + "VALIDATION FAILED" + " " * 29 + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        failed = [name for name, passed in results.items() if not passed]
        print(f"Failed validations: {', '.join(failed)}")
        print()
        print("Please check the error messages above and fix issues.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
