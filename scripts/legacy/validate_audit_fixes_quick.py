"""
DEPRECATED: Legacy quick validation script.

This file is retained for backward compatibility; please use
`scripts/validate_remediation_fixes_quick.py` for active remediation checks.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("QUICK VALIDATION OF REMEDIATION FIXES (LEGACY LOCATION)")
print("="*80)

# Test 1: Label Noise Ablation Import
print("\n[1/5] Testing label noise ablation import...")
try:
    from src.experiments.run_label_noise_ablation import (
        NoisyLabelDataset,
        LabelNoiseConfig,
        create_noisy_dataloaders,
        train_with_noisy_labels,
        run_label_noise_ablation,
        create_label_noise_summary,
        analyze_robustness_to_noise
    )
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: NoisyLabelDataset Functionality
print("\n[2/5] Testing NoisyLabelDataset...")
try:
    from torch.utils.data import TensorDataset
    
    # Create simple dataset
    data = torch.randn(100, 10)
    labels = torch.randint(0, 5, (100,))
    dataset = TensorDataset(data, labels)
    
    # Test no noise
    noisy_dataset_0 = NoisyLabelDataset(dataset, noise_rate=0.0, num_classes=5, seed=42)
    assert noisy_dataset_0.get_clean_accuracy() == 1.0, "No noise should have 100% accuracy"
    
    # Test 20% noise
    noisy_dataset_20 = NoisyLabelDataset(dataset, noise_rate=0.2, num_classes=5, seed=42)
    clean_acc = noisy_dataset_20.get_clean_accuracy()
    assert 0.79 < clean_acc < 0.81, f"20% noise should have ~80% clean accuracy, got {clean_acc}"
    
    # Test reproducibility
    noisy_dataset_20b = NoisyLabelDataset(dataset, noise_rate=0.2, num_classes=5, seed=42)
    assert np.array_equal(noisy_dataset_20.noisy_labels, noisy_dataset_20b.noisy_labels), \
        "Same seed should produce identical noise"
    
    print("   ✓ NoisyLabelDataset working correctly")
except AssertionError as e:
    print(f"   ✗ Test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")
    sys.exit(1)

# Test 3: Fairness Validation Import
print("\n[3/5] Testing fairness validation import...")
try:
    from src.utils.fairness_check import (
        TuningFairnessValidator,
        TuningFairnessError,
        TuningConfig,
        validate_tuning_fairness,
        check_tuning_parity_in_results,
        generate_fair_tuning_config
    )
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 4: Fairness Validation Functionality
print("\n[4/5] Testing fairness validation...")
try:
    # Test fair config (should pass)
    optimizers = ['SGD', 'Adam', 'SAM_SGD']
    fair_config = generate_fair_tuning_config(optimizers, n_trials=15, epochs=3)
    result = validate_tuning_fairness(optimizers, fair_config, strict=True)
    assert result == True, "Fair config should pass validation"
    
    # Test unfair config (should fail)
    unfair_config = {
        'SGD': {'n_trials': 15, 'epochs': 3, 'is_tuned': True},
        'Adam': {'n_trials': 15, 'epochs': 3, 'is_tuned': True},
        'SAM_SGD': {'n_trials': 0, 'epochs': 0, 'is_tuned': False}
    }
    
    try:
        validate_tuning_fairness(optimizers, unfair_config, strict=True)
        print("   ✗ Unfair config should have failed validation")
        sys.exit(1)
    except TuningFairnessError:
        pass  # Expected
    
    # Test permissive mode (should not raise)
    result = validate_tuning_fairness(optimizers, unfair_config, strict=False)
    assert result == False, "Unfair config should return False in permissive mode"
    
    print("   ✓ Fairness validation working correctly")
except AssertionError as e:
    print(f"   ✗ Test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")
    sys.exit(1)

# Test 5: Extended Tuning Integration
print("\n[5/5] Testing extended tuning integration...")
try:
    from run_all_kaggle import quick_tune_optimizer, get_default_hyperparameters
    
    # Check that advanced optimizers have default params
    for opt_name in ['SAM_SGD', 'SAM_Adam', 'Lookahead_SGD', 'Lookahead_Adam', 
                     'AdaBound', 'RAdam', 'LAMB']:
        params = get_default_hyperparameters(opt_name)
        assert params is not None, f"{opt_name} should have default hyperparameters"
        assert len(params) > 0, f"{opt_name} params should not be empty"
    
    print("   ✓ Extended tuning integration working correctly")
except Exception as e:
    print(f"   ✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("✅ ALL VALIDATION TESTS PASSED")
print("="*80)
print("\nRemediation fixes are working correctly. Key features validated:")
print("  ✓ Label noise ablation implementation")
print("  ✓ NoisyLabelDataset with reproducible seeding")
print("  ✓ Fairness validation utility")
print("  ✓ Extended tuning for SAM/Lookahead/AdaBound/RAdam/LAMB")
print("\nYou can now run full experiments:")
print("  python run_all_kaggle.py --experiments label_noise --quick")
print("\nFor comprehensive testing:")
print("  pytest tests/test_label_noise_ablation.py -v")
print("="*80)