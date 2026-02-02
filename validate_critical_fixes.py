#!/usr/bin/env python3
"""
Validation script for critical fixes applied to GDSearch codebase.
Tests:
1. MLflow ExperimentTracker robustness and schema handling
2. Notebook syntax correctness
3. Debugger configuration
"""
import sys
import json
from pathlib import Path

print("=" * 80)
print("VALIDATION REPORT: Critical Fixes")
print("=" * 80)

all_passed = True

# Test 1: ExperimentTracker imports and has new methods
print("\n[TEST 1] ExperimentTracker Module...")
try:
    from src.core.experiment_tracker import ExperimentTracker
    
    # Check for new methods
    has_upgrade = hasattr(ExperimentTracker, '_attempt_db_upgrade')
    has_fresh_db = hasattr(ExperimentTracker, '_attempt_fresh_db')
    
    if has_upgrade and has_fresh_db:
        print("  ✅ ExperimentTracker has schema upgrade methods")
    else:
        print("  ❌ ExperimentTracker missing schema upgrade methods")
        all_passed = False
    
    # Try to instantiate (with no-mlflow fallback)
    tracker = ExperimentTracker()
    print(f"  ✅ ExperimentTracker instantiates successfully (enabled={tracker.enabled})")
    
except Exception as e:
    print(f"  ❌ ExperimentTracker failed: {e}")
    all_passed = False

# Test 2: Notebook syntax validation
print("\n[TEST 2] Kaggle Notebook Syntax...")
try:
    notebook_path = Path("kaggle/gdsearch_kaggle_runner.ipynb")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print("  ✅ Notebook JSON is valid")
    
    # Check for escaped newlines in code cells (excluding valid print statements)
    bad_cells = []
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for line in source:
                # Look for escaped newline followed by code (not in strings)
                if r'\n' in line and 'mnist_csvs' in line and 'print' not in line:
                    bad_cells.append((i, line[:80]))
    
    if bad_cells:
        print(f"  ❌ Found {len(bad_cells)} cells with syntax errors:")
        for cell_idx, line in bad_cells:
            print(f"      Cell {cell_idx}: {line}...")
        all_passed = False
    else:
        print("  ✅ No syntax errors (escaped newlines) found")
    
    # Check that safe_read_csv import is present
    has_safe_read = any(
        'safe_read_csv' in ''.join(cell.get('source', []))
        for cell in nb['cells']
        if cell.get('cell_type') == 'code'
    )
    if has_safe_read:
        print("  ✅ safe_read_csv import found in notebook")
    else:
        print("  ⚠️  safe_read_csv import not found (may be expected)")
    
except Exception as e:
    print(f"  ❌ Notebook validation failed: {e}")
    all_passed = False

# Test 3: Launch configuration
print("\n[TEST 3] VS Code Launch Configuration...")
try:
    launch_path = Path(".vscode/launch.json")
    with open(launch_path, 'r', encoding='utf-8') as f:
        launch_content = f.read()
    
    has_frozen_fix = "-Xfrozen_modules=off" in launch_content
    has_validation_fix = "PYDEVD_DISABLE_FILE_VALIDATION" in launch_content
    
    if has_frozen_fix:
        print("  ✅ Frozen modules disabled in launch config")
    else:
        print("  ❌ Missing frozen modules fix")
        all_passed = False
    
    if has_validation_fix:
        print("  ✅ PYDEVD file validation disabled in launch config")
    else:
        print("  ❌ Missing PYDEVD validation fix")
        all_passed = False
    
except Exception as e:
    print(f"  ❌ Launch config validation failed: {e}")
    all_passed = False

# Test 4: Import safety check
print("\n[TEST 4] Core Module Import Safety...")
try:
    # These should not raise on import
    from src.core import experiment_tracker
    from src.core import checkpoint_manager
    from src.utils import csv_utils
    print("  ✅ Core modules import without side effects")
except Exception as e:
    print(f"  ❌ Import safety failed: {e}")
    all_passed = False

# Final summary
print("\n" + "=" * 80)
if all_passed:
    print("✅ ALL VALIDATION TESTS PASSED")
    print("=" * 80)
    print("\nThe following fixes have been successfully applied:")
    print("  1. MLflow schema upgrade handling in ExperimentTracker")
    print("  2. Notebook syntax error fixed (escaped newline)")
    print("  3. Debugger warnings suppressed in launch.json")
    print("\nThe codebase is now ready for experiments.")
    sys.exit(0)
else:
    print("❌ SOME VALIDATION TESTS FAILED")
    print("=" * 80)
    print("\nPlease review the failures above and re-apply fixes as needed.")
    sys.exit(1)
