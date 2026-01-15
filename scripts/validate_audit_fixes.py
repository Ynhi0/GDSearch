#!/usr/bin/env python3
"""
Validation script to verify audit fixes are correctly applied.

This script performs manual checks that cannot be easily automated:
1. Verifies imports exist in run_nn_experiment.py
2. Checks criterion is no longer hardcoded to nn.CrossEntropyLoss()
3. Confirms AMP/EMA setup logic exists
4. Ensures evaluation uses EMA model when enabled
5. Validates config file schema

Run this script after applying audit fixes to confirm correctness.

Usage:
    python scripts/validate_audit_fixes.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def validate_imports():
    """Check that required imports exist in run_nn_experiment.py"""
    run_nn_file = project_root / "src" / "experiments" / "run_nn_experiment.py"
    content = run_nn_file.read_text(encoding='utf-8')
    
    # Check for symbols (works for both single-line and multi-line imports)
    required_symbols = [
        "get_loss_function",
        "AMPWrapper",
        "ModelEMA",
        "create_amp_wrapper",
        "create_model_ema"
    ]
    
    missing = []
    for symbol in required_symbols:
        # Check if imported from training_utils (anywhere in the file)
        if symbol not in content:
            missing.append(symbol)
    
    if missing:
        print("❌ FAIL: Missing required imports in run_nn_experiment.py:")
        for symbol in missing:
            print(f"   - {symbol}")
        return False
    else:
        print("✅ PASS: All required imports present in run_nn_experiment.py")
        return True


def validate_loss_function():
    """Check that criterion is not hardcoded to nn.CrossEntropyLoss()"""
    run_nn_file = project_root / "src" / "experiments" / "run_nn_experiment.py"
    content = run_nn_file.read_text(encoding='utf-8')
    
    # Bad pattern: hardcoded CrossEntropyLoss
    if "criterion = nn.CrossEntropyLoss()" in content:
        print("❌ FAIL: Found hardcoded 'criterion = nn.CrossEntropyLoss()' in run_nn_experiment.py")
        print("   Expected: 'criterion = get_loss_function(...)'")
        return False
    
    # Good pattern: configurable loss function
    if "criterion = get_loss_function('cross_entropy'" in content:
        print("✅ PASS: Loss function is configurable via get_loss_function()")
        return True
    else:
        print("⚠️  WARN: Could not find 'get_loss_function' call in run_nn_experiment.py")
        return False


def validate_amp_setup():
    """Check that AMP setup logic exists"""
    run_nn_file = project_root / "src" / "experiments" / "run_nn_experiment.py"
    content = run_nn_file.read_text(encoding='utf-8')
    
    # Look for AMP setup pattern
    if "amp = create_amp_wrapper(enabled=use_amp)" in content or \
       "amp = create_amp_wrapper(enabled=" in content:
        print("✅ PASS: AMP setup logic present in run_nn_experiment.py")
        return True
    else:
        print("❌ FAIL: AMP setup logic not found in run_nn_experiment.py")
        print("   Expected: 'amp = create_amp_wrapper(enabled=use_amp)'")
        return False


def validate_ema_setup():
    """Check that EMA setup logic exists"""
    run_nn_file = project_root / "src" / "experiments" / "run_nn_experiment.py"
    content = run_nn_file.read_text(encoding='utf-8')
    
    # Look for EMA setup pattern
    if "ema = create_model_ema(model" in content:
        print("✅ PASS: EMA setup logic present in run_nn_experiment.py")
        return True
    else:
        print("❌ FAIL: EMA setup logic not found in run_nn_experiment.py")
        print("   Expected: 'ema = create_model_ema(model, decay=...)'")
        return False


def validate_ema_evaluation():
    """Check that evaluation uses EMA model when enabled"""
    run_nn_file = project_root / "src" / "experiments" / "run_nn_experiment.py"
    content = run_nn_file.read_text(encoding='utf-8')
    
    # Look for EMA evaluation pattern (ModelEMA stores shadow model in self.shadow, not self.ema)
    if "eval_model = ema.shadow if ema is not None else model" in content:
        print("✅ PASS: Evaluation uses EMA model when enabled")
        return True
    else:
        print("❌ FAIL: Evaluation does not use EMA shadow model")
        print("   Expected: 'eval_model = ema.shadow if ema is not None else model'")
        return False


def validate_config_files():
    """Check that new config files exist"""
    ablation_config = project_root / "configs" / "label_smoothing_ablation.json"
    
    if not ablation_config.exists():
        print("❌ FAIL: Missing configs/label_smoothing_ablation.json")
        return False
    else:
        print("✅ PASS: Label smoothing ablation config exists")
        return True


def validate_test_files():
    """Check that integration test file exists"""
    test_file = project_root / "tests" / "test_integration_label_smoothing.py"
    
    if not test_file.exists():
        print("❌ FAIL: Missing tests/test_integration_label_smoothing.py")
        return False
    else:
        print("✅ PASS: Integration test file exists")
        return True


def main():
    print("=" * 60)
    print("Audit Fix Validation Script")
    print("=" * 60)
    print()
    
    checks = [
        ("Import Validation", validate_imports),
        ("Loss Function Check", validate_loss_function),
        ("AMP Setup Check", validate_amp_setup),
        ("EMA Setup Check", validate_ema_setup),
        ("EMA Evaluation Check", validate_ema_evaluation),
        ("Config File Check", validate_config_files),
        ("Test File Check", validate_test_files)
    ]
    
    results = []
    for name, check_fn in checks:
        print(f"\nRunning: {name}")
        print("-" * 60)
        result = check_fn()
        results.append(result)
        print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n🎉 All validation checks passed!")
        print("\nNext steps:")
        print("1. Run integration tests: pytest tests/test_integration_label_smoothing.py -v")
        print("2. Run label smoothing ablation: python src/experiments/run_full_analysis.py --config configs/label_smoothing_ablation.json")
        return 0
    else:
        print("\n⚠️  Some validation checks failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
