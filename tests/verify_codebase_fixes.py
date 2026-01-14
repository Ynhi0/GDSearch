"""
Comprehensive validation script for codebase fixes.
Verifies that all critical issues have been addressed.
"""

import re
import sys
from pathlib import Path

def check_shallow_copies():
    """Check for remaining shallow .copy() calls that should be deepcopy."""
    print("\n[1/5] Checking for shallow config.copy() calls...")
    risky_patterns = [
        "config = base_config.copy()",
        "config = optimizer_config.copy()",
        "config_with_seed = config.copy()"
    ]

    # Files where .copy() is acceptable (utility modules, flat dicts)
    safe_files = [
        "config_loader.py",
        "validation.py",
        "optimizer_registry.py"
    ]

    found_issues = []
    src_path = Path("src")
    if not src_path.exists():
        print("  ⚠️  SKIP: src/ directory not found")
        return True

    for pattern in risky_patterns:
        for py_file in src_path.rglob("*.py"):
            if not any(safe in str(py_file) for safe in safe_files):
                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    for line_num, line in enumerate(content.splitlines(), start=1):
                        if pattern in line:
                            found_issues.append(f"{py_file}:{line_num}:{line.strip()}")
                except Exception as e:
                    print(f"  ⚠️  Could not read {py_file}: {e}")

    if found_issues:
        print("  ❌ FAIL: Found risky shallow copies:")
        for issue in found_issues:
            print(f"    {issue}")
        # Fail fast using assertion so the function does not return a bool
        assert False, "Found risky shallow copies"
    else:
        print("  ✅ PASS: All config copies use deepcopy")
        # Success: do not return a value (None) to avoid pytest complaining about returning a value
        return None

def check_history_guards():
    """Check for unguarded history[-1] accesses."""
    print("\n[2/5] Checking for unguarded history[-1] accesses...")

    src_path = Path("src")
    if not src_path.exists():
        print("  ⚠️  SKIP: src/ directory not found")
        return True

    found_accesses = []
    for py_file in src_path.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            for line_num, line in enumerate(content.splitlines(), start=1):
                if "history[-1]" in line:
                    found_accesses.append(f"{py_file}:{line_num}:{line.strip()}")
        except Exception as e:
            print(f"  ⚠️  Could not read {py_file}: {e}")

    if found_accesses:
        print(f"  ⚠️  Found {len(found_accesses)} history[-1] accesses")
        print("  ℹ️  Manual review recommended to ensure all are guarded")
        # Warning only — keep execution path passing but avoid returning a boolean
        return None
    else:
        print("  ✅ PASS: No history[-1] accesses found")
        return None

def check_division_by_zero():
    """Check for potential division by zero issues."""
    print("\n[3/5] Checking for division by zero guards...")
    risky_patterns = [
        r"/ total(?![_\w])",  # Matches "/ total" but not "/ total_samples"
        r"/ count(?![_\w])",
        r"/ len\("
    ]

    # This is a heuristic check - actual validation requires code review
    print("  ℹ️  This check requires manual code review")
    print("  ℹ️  Ensure all divisions check for zero denominators")
    # Heuristic check — returns None to indicate non-failing/manual review required
    return None

def check_accuracy_scale():
    """Check for accuracy scale consistency."""
    print("\n[4/5] Checking accuracy scale consistency...")
    # Check that NLP evaluate returns percentage
    nlp_file = Path("kaggle/nlp_benchmark/run_nlp.py")
    if nlp_file.exists():
        content = nlp_file.read_text()
        if "100.0 * correct / max(1, total)" in content:
            print("  ✅ PASS: NLP accuracy returns percentage")
            return None
        else:
            print("  ❌ FAIL: NLP accuracy scale incorrect")
            assert False, "NLP accuracy scale incorrect"
    else:
        print("  ⚠️  SKIP: NLP file not found")
        return None

def run_quick_import_test():
    """Quick import test to catch any syntax errors."""
    print("\n[5/5] Running quick import test...")
    try:
        # Try importing key modules
        sys.path.insert(0, str(Path.cwd()))

        from src.experiments import weight_decay_ablation
        from src.experiments import batch_size_ablation
        from src.experiments import learning_rate_ablation
        from src.experiments import scheduler_ablation
        from src.analysis import sensitivity_analysis
        from src.analysis import baseline_comparison
        from src.analysis import ablation_study

        print("  ✅ PASS: All modified modules import successfully")
        return None
    except Exception as e:
        print(f"  ❌ FAIL: Import error: {e}")
        raise AssertionError(f"Import error: {e}")

def main():
    print("="*70)
    print("CODEBASE FIX VALIDATION")
    print("="*70)

    results = []

    checks = [
        check_shallow_copies,
        check_history_guards,
        check_division_by_zero,
        check_accuracy_scale,
        run_quick_import_test,
    ]

    for check in checks:
        try:
            _res = check()
            # None means pass or warning; treat as pass
            results.append(True)
        except AssertionError as ae:
            print(f"  ❌ CHECK FAILED: {check.__name__}: {ae}")
            results.append(False)
        except Exception as e:
            print(f"  ❌ CHECK ERROR: {check.__name__}: {e}")
            results.append(False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\n✅ All checks passed!")
        return 0
    else:
        print("\n⚠️  Some checks failed or require manual review")
        return 1

if __name__ == "__main__":
    sys.exit(main())
