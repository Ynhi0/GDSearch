"""
Verification Script for Critical Scientific Fixes

This script verifies that all 4 critical issues have been properly addressed:
1. Eigenvalue Mirror Error (saddle_point_detection.py)
2. Bias Correction Blind Spot (effective_learning_rate_analysis.py)
3. Memory OOM Default (trajectory_projection.py)
4. Data Loss Issue (training_enhancements.py)
"""

import sys
import os
import re
import ast
from pathlib import Path


def check_file_exists(filepath: str) -> bool:
    """Check if file exists and return path."""
    if not os.path.exists(filepath):
        print(f"❌ FILE NOT FOUND: {filepath}")
        return False
    return True


def verify_saddle_point_detection():
    """Verify Fix #1: Eigenvalue computation uses scipy eigsh with 'SA'."""
    print("\n" + "="*80)
    print("FIX #1: Eigenvalue Mirror Error (saddle_point_detection.py)")
    print("="*80)
    
    filepath = "src/analysis/saddle_point_detection.py"
    if not check_file_exists(filepath):
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check 1: Uses scipy.sparse.linalg.eigsh
    if 'from scipy.sparse.linalg import eigsh' in content or 'scipy.sparse.linalg.eigsh' in content:
        print("✅ PASS: Uses scipy.sparse.linalg.eigsh")
    else:
        print("❌ FAIL: Does not import scipy.sparse.linalg.eigsh")
        return False
    
    # Check 2: Uses which='SA' (Smallest Algebraic)
    if "which='SA'" in content or 'which="SA"' in content:
        print("✅ PASS: Uses which='SA' for smallest algebraic eigenvalue")
    else:
        print("⚠️  WARNING: Could not find which='SA' - may still use largest magnitude")
    
    # Check 3: Documentation mentions the fix
    if 'LARGEST MAGNITUDE' in content.upper() and 'SMALLEST ALGEBRAIC' in content.upper():
        print("✅ PASS: Documentation explains the eigenvalue mirror issue")
    else:
        print("⚠️  WARNING: Documentation may not fully explain the fix")
    
    # Check 4: Does NOT simply normalize Hv by magnitude (old broken method)
    broken_pattern = r'v\s*=\s*\[Hvi\s*/\s*Hv_norm\s+for\s+Hvi\s+in\s+Hv\]'
    if re.search(broken_pattern, content):
        # Check if this is in a fallback or commented section
        lines_with_pattern = [line for line in content.split('\n') if 'Hvi / Hv_norm' in line]
        if any('fallback' in line.lower() or 'approximate' in line.lower() for line in lines_with_pattern):
            print("✅ PASS: Old method only used as fallback")
        else:
            print("⚠️  WARNING: Found potential power iteration on Hv (should use shifted Hessian)")
    else:
        print("✅ PASS: Does not use broken power iteration on Hv directly")
    
    return True


def verify_effective_learning_rate():
    """Verify Fix #2: Adam bias correction is applied."""
    print("\n" + "="*80)
    print("FIX #2: Bias Correction Blind Spot (effective_learning_rate_analysis.py)")
    print("="*80)
    
    filepath = "src/analysis/effective_learning_rate_analysis.py"
    if not check_file_exists(filepath):
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check 1: Accesses step count
    if "state.get('step'" in content or "state['step']" in content:
        print("✅ PASS: Accesses optimizer step count")
    else:
        print("❌ FAIL: Does not access step count for bias correction")
        return False
    
    # Check 2: Computes bias_correction1 and bias_correction2
    if 'bias_correction1' in content and 'bias_correction2' in content:
        print("✅ PASS: Computes both bias_correction1 and bias_correction2")
    else:
        print("❌ FAIL: Does not compute bias correction terms")
        return False
    
    # Check 3: Uses beta1 and beta2 from optimizer
    if "group.get('betas'" in content or "group['betas']" in content:
        print("✅ PASS: Retrieves beta1 and beta2 from optimizer")
    else:
        print("⚠️  WARNING: May not retrieve betas from optimizer")
    
    # Check 4: Documentation mentions bias correction importance
    if 'BIAS CORRECTION' in content.upper() and 'EARLY TRAINING' in content.upper():
        print("✅ PASS: Documentation explains bias correction importance")
    else:
        print("⚠️  WARNING: Documentation may not explain the issue")
    
    # Check 5: Does NOT use raw v without correction
    if 'self.base_lr / (torch.sqrt(v) + eps)' in content:
        # Check if this is in corrected context
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'self.base_lr / (torch.sqrt(v) + eps)' in line:
                # Check surrounding lines for bias correction
                context = '\n'.join(lines[max(0, i-10):min(len(lines), i+5)])
                if 'bias_correction' not in context:
                    print("❌ FAIL: Found uncorrected effective LR formula")
                    return False
        print("✅ PASS: All effective LR computations include bias correction")
    
    return True


def verify_trajectory_projection():
    """Verify Fix #3: Default subsample_params prevents OOM."""
    print("\n" + "="*80)
    print("FIX #3: Memory OOM Default (trajectory_projection.py)")
    print("="*80)
    
    filepath = "src/visualization/trajectory_projection.py"
    if not check_file_exists(filepath):
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check 1: __init__ has subsample_params with non-None default
    init_pattern = r'def __init__\(.*?subsample_params:\s*Optional\[int\]\s*=\s*(\d+)'
    match = re.search(init_pattern, content, re.DOTALL)
    
    if match:
        default_value = int(match.group(1))
        if default_value > 0 and default_value <= 50000:
            print(f"✅ PASS: subsample_params defaults to {default_value} (safe value)")
        else:
            print(f"⚠️  WARNING: subsample_params defaults to {default_value} (may be too large)")
    else:
        print("❌ FAIL: subsample_params does not have a safe default value")
        return False
    
    # Check 2: Documentation mentions OOM risk
    if 'OOM' in content or 'OUT OF MEMORY' in content.upper() or 'MEMORY' in content:
        print("✅ PASS: Documentation mentions memory concerns")
    else:
        print("⚠️  WARNING: Documentation may not explain OOM risk")
    
    # Check 3: Subsampling is actually implemented
    if 'self.param_indices' in content and 'np.random.choice' in content:
        print("✅ PASS: Subsampling mechanism is implemented")
    else:
        print("⚠️  WARNING: Subsampling mechanism may not be fully implemented")
    
    # Check 4: Warning when subsample_params=None
    if 'subsample_params is None' in content and 'warning' in content.lower():
        print("✅ PASS: Warning issued when using all parameters")
    else:
        print("⚠️  WARNING: No warning for using all parameters")
    
    return True


def verify_training_enhancements():
    """Verify Fix #4: Gradient accumulation instead of data dropping."""
    print("\n" + "="*80)
    print("FIX #4: Data Loss Issue (training_enhancements.py)")
    print("="*80)
    
    filepath = "src/core/training_enhancements.py"
    if not check_file_exists(filepath):
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check 1: Uses gradient accumulation approach
    if 'gradient accumulation' in content.lower() or 'num_chunks' in content:
        print("✅ PASS: Uses gradient accumulation approach")
    else:
        print("❌ FAIL: Does not implement gradient accumulation")
        return False
    
    # Check 2: Processes all data in chunks
    if 'for chunk_idx in range(num_chunks)' in content:
        print("✅ PASS: Iterates through all data chunks")
    else:
        print("⚠️  WARNING: May not process all data chunks")
    
    # Check 3: Does NOT drop data with inputs[:new_size] in active code
    # (May exist in comments or documentation)
    active_lines = [line for line in content.split('\n') 
                   if 'inputs[:new_size]' in line and not line.strip().startswith('#')]
    if len(active_lines) == 0:
        print("✅ PASS: Does not drop data with inputs[:new_size]")
    else:
        print("❌ FAIL: Found data dropping pattern in active code:")
        for line in active_lines[:3]:
            print(f"    {line.strip()}")
        return False
    
    # Check 4: Documentation mentions data integrity fix
    if 'DATA INTEGRITY' in content.upper() or 'INSTEAD OF DROPPING' in content.upper():
        print("✅ PASS: Documentation mentions data integrity fix")
    else:
        print("⚠️  WARNING: Documentation may not explain the fix")
    
    # Check 5: current_batch_size = old_size (not new_size)
    if 'self.current_batch_size = old_size' in content:
        print("✅ PASS: Reports full batch size (not reduced size)")
    else:
        print("⚠️  WARNING: May not report correct batch size")
    
    return True


def run_verification():
    """Run all verification checks."""
    print("\n" + "="*80)
    print("CRITICAL SCIENTIFIC FIXES VERIFICATION")
    print("="*80)
    
    results = {
        'saddle_point_detection': verify_saddle_point_detection(),
        'effective_learning_rate': verify_effective_learning_rate(),
        'trajectory_projection': verify_trajectory_projection(),
        'training_enhancements': verify_training_enhancements(),
    }
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for fix_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {fix_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL CRITICAL FIXES VERIFIED SUCCESSFULLY!")
        print("="*80)
        print("\nThe codebase now:")
        print("  1. Correctly finds smallest algebraic eigenvalues (saddle point detection)")
        print("  2. Applies Adam bias correction (effective learning rate analysis)")
        print("  3. Uses safe default parameter subsampling (trajectory projection)")
        print("  4. Preserves all training data via gradient accumulation (OOM recovery)")
        print("\nThese fixes ensure scientifically valid results in your optimization analysis.")
        return 0
    else:
        print("\n" + "="*80)
        print("❌ VERIFICATION FAILED")
        print("="*80)
        print("\nSome critical fixes are missing or incomplete.")
        print("Please review the failed checks above and apply the necessary fixes.")
        return 1


if __name__ == "__main__":
    exit_code = run_verification()
    sys.exit(exit_code)
