#!/usr/bin/env python3
"""Quick validation test for configuration and validation logic fixes."""

import sys
from run_all_kaggle import (
    validate_learning_rate,
    validate_optimizer_name,
    safe_config_int,
    safe_config_float,
    safe_config_bool
)

def test_learning_rate_validation():
    """Test Fix 4: Learning rate validation."""
    print("\n[TEST] Learning Rate Validation")
    
    # Valid cases
    try:
        validate_learning_rate(0.01, 'SGD')
        validate_learning_rate(0.001, 'Adam')
        print("  ✓ Valid LR values accepted")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Invalid: too large
    try:
        validate_learning_rate(15.0, 'SGD')
        print("  ✗ FAILED: Should reject LR > 10")
        return False
    except ValueError:
        print("  ✓ Correctly rejected LR > 10")
    
    # Invalid: negative
    try:
        validate_learning_rate(-0.1, 'SGD')
        print("  ✗ FAILED: Should reject negative LR")
        return False
    except ValueError:
        print("  ✓ Correctly rejected negative LR")
    
    # Invalid: wrong type
    try:
        validate_learning_rate("not_a_number", 'SGD')
        print("  ✗ FAILED: Should reject non-numeric LR")
        return False
    except TypeError:
        print("  ✓ Correctly rejected non-numeric LR")
    
    return True


def test_optimizer_name_validation():
    """Test Fix 8: Optimizer name validation."""
    print("\n[TEST] Optimizer Name Validation")
    
    # Valid cases
    try:
        result = validate_optimizer_name('adam')
        assert result == 'adam', f"Expected 'adam', got '{result}'"
        result = validate_optimizer_name('SGD')
        assert result == 'sgd', f"Expected 'sgd', got '{result}'"
        result = validate_optimizer_name('AdamW')
        assert result == 'adamw', f"Expected 'adamw', got '{result}'"
        print("  ✓ Valid optimizer names accepted and normalized")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Invalid: typo should suggest correction
    try:
        validate_optimizer_name('adm')  # typo for adam
        print("  ✗ FAILED: Should reject invalid optimizer name")
        return False
    except ValueError as e:
        error_msg = str(e)
        if 'adam' in error_msg.lower():
            print(f"  ✓ Correctly rejected typo with suggestion")
        else:
            print(f"  ⚠ Rejected but no helpful suggestion: {error_msg[:80]}")
    
    return True


def test_safe_config_conversions():
    """Test Fix 6: Safe config type conversions."""
    print("\n[TEST] Safe Config Type Conversions")
    
    # Test int conversion from string
    try:
        config = {'epochs': '50', 'batch_size': 128}
        epochs = safe_config_int(config, 'epochs', 10)
        assert epochs == 50 and isinstance(epochs, int)
        batch_size = safe_config_int(config, 'batch_size', 64)
        assert batch_size == 128 and isinstance(batch_size, int)
        print("  ✓ Integer conversion from string works")
    except Exception as e:
        print(f"  ✗ FAILED int conversion: {e}")
        return False
    
    # Test float conversion from string
    try:
        config = {'lr': '0.001', 'weight_decay': 0.0001}
        lr = safe_config_float(config, 'lr', 0.01)
        assert lr == 0.001 and isinstance(lr, float)
        wd = safe_config_float(config, 'weight_decay', 0.0)
        assert wd == 0.0001 and isinstance(wd, float)
        print("  ✓ Float conversion from string works")
    except Exception as e:
        print(f"  ✗ FAILED float conversion: {e}")
        return False
    
    # Test bool conversion from various formats
    try:
        config = {
            'use_amp': 'true',
            'use_ema': '1',
            'deterministic': 'yes',
            'skip_tuning': False
        }
        assert safe_config_bool(config, 'use_amp', False) == True
        assert safe_config_bool(config, 'use_ema', False) == True
        assert safe_config_bool(config, 'deterministic', False) == True
        assert safe_config_bool(config, 'skip_tuning', True) == False
        print("  ✓ Boolean conversion from strings works")
    except Exception as e:
        print(f"  ✗ FAILED bool conversion: {e}")
        return False
    
    # Test invalid conversion
    try:
        config = {'epochs': 'not_a_number'}
        safe_config_int(config, 'epochs', 10)
        print("  ✗ FAILED: Should reject invalid int string")
        return False
    except ValueError:
        print("  ✓ Correctly rejected invalid int conversion")
    
    return True


def main():
    """Run all validation tests."""
    print("="*70)
    print("Configuration and Validation Logic Fixes - Verification")
    print("="*70)
    
    results = []
    
    results.append(("Learning Rate Validation", test_learning_rate_validation()))
    results.append(("Optimizer Name Validation", test_optimizer_name_validation()))
    results.append(("Safe Config Conversions", test_safe_config_conversions()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All validation fixes verified successfully!")
        return 0
    else:
        print("\n✗ Some tests failed - review implementation")
        return 1


if __name__ == '__main__':
    sys.exit(main())
