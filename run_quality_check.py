#!/usr/bin/env python3
"""
Comprehensive codebase quality check script.
Scans for common issues and validates all critical modules.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.stderr and not result.stderr.startswith("Successfully"):
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("="*80)
    print("COMPREHENSIVE CODEBASE QUALITY CHECK")
    print("="*80)
    
    results = {}
    
    # Test 1: Import all critical modules
    results['imports'] = run_command(
        'python -c "from src.core.optimizers import *; from src.core.test_functions import *; from src.utils.config_loader import *; print(\'[OK] All critical imports successful\')"',
        "TEST 1: Import Verification"
    )
    
    # Test 2: Verify config loader
    results['config_loader'] = run_command(
        'python -c "from src.utils.config_loader import load_optimizer_config; cfg = load_optimizer_config(\'benchmark_hyperparameters\', \'resnet_cifar10\', \'Adam\'); print(f\'[OK] Loaded Adam config: lr={cfg[\\\'lr\\\']}\')"',
        "TEST 2: Config Loader Verification"
    )
    
    # Test 3: Pyright type checking
    results['pyright'] = run_command(
        'pyright src/core/optimizers.py src/utils/config_loader.py src/core/test_functions.py',
        "TEST 3: Pyright Type Checking"
    )
    
    # Test 4: Verify Adam/AdamW weight decay
    results['weight_decay'] = run_command(
        'python -c "from src.core.optimizers import Adam, AdamW; a1=Adam(lr=0.01, weight_decay=0.01); a2=AdamW(lr=0.01, weight_decay=0.01); print(f\'[OK] Adam L2: {a1.weight_decay}, AdamW decoupled: {a2.weight_decay}\')"',
        "TEST 4: Weight Decay Implementation"
    )
    
    # Test 5: Check for runtime errors in main orchestrator
    results['orchestrator'] = run_command(
        'python -c "import sys; sys.path.insert(0, \'.\'); from runners.run_all_kaggle import run_sam_sensitivity; print(\'[OK] Main orchestrator imports successfully\')"',
        "TEST 5: Main Orchestrator Import"
    )
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL QUALITY CHECKS PASSED")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
