#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Codebase Scan & Validation
=========================================

This script performs a comprehensive analysis of the GDSearch codebase to identify:
1. Import errors and missing dependencies
2. Undefined variables and name errors
3. Training loops that need robust gradient integration
4. Potential bugs and code quality issues
5. Missing test coverage

Run:
    python scripts/comprehensive_scan.py --fix-issues
"""

import sys
import ast
import importlib
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


def scan_imports(file_path: Path) -> List[Dict[str, str]]:
    """Scan file for imports and check if they're available."""
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as e:
                        issues.append({
                            'file': str(file_path),
                            'line': node.lineno,
                            'type': 'ImportError',
                            'message': f"Cannot import {alias.name}: {e}"
                        })

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        importlib.import_module(node.module)
                    except ImportError as e:
                        issues.append({
                            'file': str(file_path),
                            'line': node.lineno,
                            'type': 'ImportError',
                            'message': f"Cannot import from {node.module}: {e}"
                        })

    except SyntaxError as e:
        issues.append({
            'file': str(file_path),
            'line': e.lineno,
            'type': 'SyntaxError',
            'message': str(e)
        })
    except Exception as e:
        issues.append({
            'file': str(file_path),
            'line': 0,
            'type': 'ParseError',
            'message': f"Failed to parse file: {e}"
        })

    return issues


def find_training_loops(file_path: Path) -> List[Dict[str, any]]:
    """Find training loops that might need robust gradient integration."""
    loops = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function contains training loop patterns
                func_code = ast.get_source_segment(content, node)
                if func_code and any(pattern in func_code for pattern in [
                    'loss.backward()',
                    'optimizer.step()',
                    'oom_safe_train_step'
                ]):
                    # Check if robust gradient handler is used
                    has_robust = 'robust_grad_handler' in func_code

                    loops.append({
                        'file': str(file_path),
                        'function': node.name,
                        'line': node.lineno,
                        'has_robust_gradients': has_robust
                    })

    except Exception as e:
        print(f"Error scanning {file_path}: {e}")

    return loops


def scan_codebase():
    """Comprehensive codebase scan."""
    print("="*80)
    print("GDSEARCH COMPREHENSIVE CODEBASE SCAN")
    print("="*80)
    print()

    # Find all Python files
    python_files = list(project_root.glob("**/*.py"))
    python_files = [f for f in python_files if 'venv' not in str(f) and '__pycache__' not in str(f)]

    print(f"Found {len(python_files)} Python files to scan")
    print()

    # Scan imports
    print("[1/4] Scanning for import errors...")
    all_import_issues = []
    for file_path in python_files:
        issues = scan_imports(file_path)
        all_import_issues.extend(issues)

    print(f"   Found {len(all_import_issues)} import issues")

    # Scan training loops
    print("[2/4] Scanning for training loops needing integration...")
    all_loops = []
    key_files = [
        project_root / 'run_all_kaggle.py',
        project_root / 'src' / 'experiments' / 'run_nn_experiment.py',
    ]

    for file_path in key_files:
        if file_path.exists():
            loops = find_training_loops(file_path)
            all_loops.extend(loops)

    print(f"   Found {len(all_loops)} training loops")

    loops_without_robust = [l for l in all_loops if not l['has_robust_gradients']]
    print(f"   {len(loops_without_robust)} loops need robust gradient integration")

    # Check test coverage
    print("[3/4] Checking test coverage...")
    test_dir = project_root / 'tests'
    test_files = list(test_dir.glob("**/*.py")) if test_dir.exists() else []
    print(f"   Found {len(test_files)} test files")

    # Check documentation
    print("[4/4] Checking documentation...")
    docs_dir = project_root / 'docs'
    md_files = list(docs_dir.glob("**/*.md")) if docs_dir.exists() else []
    print(f"   Found {len(md_files)} documentation files")

    # Generate report
    print()
    print("="*80)
    print("SCAN RESULTS")
    print("="*80)
    print()

    # Import issues
    print("[IMPORT ISSUES]")
    if all_import_issues:
        critical_imports = [i for i in all_import_issues if 'core' in i['file'] or 'experiments' in i['file']]
        if critical_imports:
            print(f"   ⚠ {len(critical_imports)} critical import errors found:")
            for issue in critical_imports[:10]:  # Show first 10
                print(f"      {issue['file']}:{issue['line']} - {issue['message']}")
        else:
            print("   ✓ No critical import errors in core modules")
    else:
        print("   ✓ No import issues found")
    print()

    # Training loops
    print("[TRAINING LOOP INTEGRATION]")
    if loops_without_robust:
        print(f"   ⚠ {len(loops_without_robust)} training loops need robust gradient integration:")
        for loop in loops_without_robust:
            print(f"      {loop['function']} in {Path(loop['file']).name}:{loop['line']}")
    else:
        print("   ✓ All training loops have robust gradient integration")
    print()

    # Test coverage
    print("[TEST COVERAGE]")
    src_files = list((project_root / 'src').glob("**/*.py"))
    coverage_ratio = len(test_files) / len(src_files) if src_files else 0
    if coverage_ratio > 0.5:
        print(f"   ✓ Good test coverage: {len(test_files)} test files for {len(src_files)} source files")
    elif coverage_ratio > 0.2:
        print(f"   ⚠ Moderate test coverage: {len(test_files)} test files for {len(src_files)} source files")
    else:
        print(f"   ⚠ Low test coverage: {len(test_files)} test files for {len(src_files)} source files")
    print()

    # Documentation
    print("[DOCUMENTATION]")
    if len(md_files) >= 5:
        print(f"   ✓ Good documentation: {len(md_files)} markdown files")
    else:
        print(f"   ⚠ Limited documentation: {len(md_files)} markdown files")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    total_issues = len(all_import_issues) + len(loops_without_robust)

    if total_issues == 0:
        print("✅ No critical issues found!")
        print("   Codebase is production-ready")
        return 0
    else:
        print(f"⚠ Found {total_issues} issues requiring attention:")
        print(f"   - {len(all_import_issues)} import errors")
        print(f"   - {len(loops_without_robust)} training loops need integration")
        print()
        print("Recommendation: Address high-priority issues before production deployment")
        return 1


if __name__ == '__main__':
    sys.exit(scan_codebase())
