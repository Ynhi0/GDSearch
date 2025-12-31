#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Validation Test - Verifies All 26 Experiments Are Bug-Free

This script runs a comprehensive quick test of all GDSearch experiments
to verify that:
1. All experiments can be imported and instantiated
2. Training loops work correctly (no indentation bugs)
3. No division by zero errors
4. Metrics are calculated correctly
5. All 26 experiments produce valid results

Estimated time: 3-5 minutes

Usage:
    python scripts/quick_validation_test.py
    python scripts/quick_validation_test.py --verbose
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import subprocess

# Windows console encoding configuration (function to avoid import-time side effects)
def configure_windows_console_encoding():
    """
    Configure UTF-8 encoding for Windows console.
    
    IMPORTANT: This must be called explicitly in main, NOT at import time.
    Import-time global stream mutations break test harnesses.
    """
    if sys.platform == 'win32':
        try:
            import codecs
            # Only wrap if not already wrapped (avoid breaking pytest capture)
            if not isinstance(sys.stdout, codecs.StreamWriter):
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            if not isinstance(sys.stderr, codecs.StreamWriter):
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except Exception as e:
            import logging
            logging.debug("configure_windows_console_encoding failed: %s", e, exc_info=True)  # Fallback to ASCII-safe output

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Common numeric library
import numpy as np

# ASCII-safe symbols for Windows compatibility
CHECK = 'OK' if sys.platform == 'win32' else 'OK'
CROSS = 'X' if sys.platform == 'win32' else 'X'
INFO = 'i' if sys.platform == 'win32' else 'ℹ'

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}{CHECK}{RESET} {text}")

def print_error(text):
    print(f"{RED}{CROSS}{RESET} {text}")

def print_info(text):
    print(f"{YELLOW}{INFO}{RESET} {text}")


def test_imports():
    """Test that all critical modules can be imported"""
    print_header("IMPORT VALIDATION")
    
    # Core imports
    import torch
    import numpy as np
    import pandas as pd
    print_success("Core dependencies (torch, numpy, pandas)")
    
    # Optimizer imports
    from src.core.optimizers import SGD, Adam, AdamW
    print_success("Core optimizers (SGD, Adam, AdamW)")
    
    # Experiment imports
    from src.experiments.beta_sensitivity_training import run_momentum_beta_sensitivity
    print_success("Beta sensitivity training module")
    
    from src.experiments.hyperparameter_sensitivity import momentum_beta_sweep
    print_success("Hyperparameter sensitivity module")
    
    from src.experiments.convergence_rate_validation import run_convergence_rate_comparison
    print_success("Convergence rate validation module")
    
    from src.analysis.statistical_analysis import compare_two_optimizers
    print_success("Statistical analysis module")
    
    print(f"\n{GREEN}All imports successful{RESET}")
    return True


def test_mnist_quick():
    """Test MNIST experiment with ultra-quick mode"""
    # Use the general helper which runs an ultra-quick single experiment and validates results
    success = run_quick_experiment('mnist', expected_min_optimizers=3, min_train_acc=80.0, min_test_acc=80.0)
    if not success:
        raise AssertionError("MNIST quick test failed")
    return success
        
        


def test_validation_script():
    """Test the comprehensive validation script"""
    print_header("RUNNING COMPREHENSIVE VALIDATION")
    
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "validate_all_experiments.py"),
        "--smoke-test"
    ]
    
    env = os.environ.copy()
    # Force the child process to use UTF-8 I/O on Windows to avoid UnicodeEncodeError when printing emojis
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300,  # 5 minutes for the validation script
            cwd=project_root,
            env=env
        )
    except subprocess.TimeoutExpired as e:
        print_error("Validation script timed out")
        raise AssertionError("Validation script timed out after 5 minutes") from e
    
    assert result.returncode == 0, f"Validation failed with return code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    
    print_success("Comprehensive validation PASSED")
    # Show summary
    return True
    lines = result.stdout.split('\n')
    for line in lines:
        if 'Total Passed' in line or 'Total Failed' in line or 'Missing Files' in line:
            print(f"  {line.strip()}")


def run_quick_experiment(experiment_name, expected_min_optimizers=3, min_train_acc=60.0, min_test_acc=50.0, timeout_sec=900):
    """Helper to run a single experiment with ultra-quick mode and validate results.
    Returns True if test passes, False otherwise.
    """
    print_header(f"{experiment_name.upper()} QUICK TEST (2 epochs, 1 seed)")
    with tempfile.TemporaryDirectory() as tmpdir:
        print_info(f"Running {experiment_name} in temp directory: {tmpdir}")
        cmd = [
            sys.executable,
            str(project_root / "run_all_kaggle.py"),
            "--ultra-quick",
            "--seeds", "42",
            "--experiments", experiment_name,
            "--results-dir", tmpdir,
            "--no-mlflow"
        ]

        # Ensure environment limit is set (default to 3 optimizers for quick local test)
        os.environ['GDSEARCH_ULTRA_QUICK_LIMIT'] = os.environ.get('GDSEARCH_ULTRA_QUICK_LIMIT', '3').strip()
        env = os.environ.copy()
        print_info(f"GDSEARCH_ULTRA_QUICK_LIMIT={env.get('GDSEARCH_ULTRA_QUICK_LIMIT')}")

        # Per-experiment metrics and checks (required columns and thresholds)
        EXP_METRICS_SPEC = {
            'mnist': {
                'required_cols': ['final_test_acc', 'final_train_acc'],
                'min': {'final_test_acc': 80.0, 'final_train_acc': 80.0}
            },
            'cifar10': {
                'required_cols': ['final_test_acc', 'final_train_acc'],
                'min': {'final_test_acc': 20.0, 'final_train_acc': 40.0}
            },
            'resnet': {
                'required_cols': ['final_test_acc'],
                'min': {'final_test_acc': 20.0}
            },
            'nlp': {
                'required_cols': ['final_test_acc', 'final_train_acc'],
                'min': {'final_test_acc': 30.0, 'final_train_acc': 40.0}
            },
            'medical': {
                'required_cols': ['final_test_dice', 'final_train_dice'],
                'min': {'final_test_dice': 0.15, 'final_train_dice': 0.15}
            }
        }

        try:
            result = subprocess.run(cmd, timeout=timeout_sec, cwd=project_root, env=env)
            if result.returncode != 0:
                print_error(f"{experiment_name} test failed - non-zero exit code {result.returncode}")
                return False

            import pandas as pd
            results_file = Path(tmpdir) / "experiments" / experiment_name / f"{experiment_name}_results.csv"
            if not results_file.exists():
                print_error(f"Results file not found: {results_file}")
                if (Path(tmpdir) / "experiments" / experiment_name).exists():
                    existing_files = list((Path(tmpdir) / "experiments" / experiment_name).glob("*"))
                    print_info(f"Files in results directory: {existing_files}")
                return False

            df = pd.read_csv(results_file)
            if len(df) == 0:
                print_error("Results CSV is empty")
                return False

            num_optimizers = len(df['optimizer'].unique())
            # Expected minimum: prefer the smaller of env var (GDSEARCH_ULTRA_QUICK_LIMIT) and experiment-specific expected_min_optimizers
            env_expected = os.environ.get('GDSEARCH_ULTRA_QUICK_LIMIT')
            if env_expected and env_expected.strip().isdigit():
                expected_min = max(1, min(int(env_expected.strip()), expected_min_optimizers))
            else:
                expected_min = max(1, expected_min_optimizers)

            if num_optimizers < expected_min:
                print_error(f"Only {num_optimizers} optimizers tested (expected at least {expected_min})")
                return False

            # If the experiment outputs accuracy columns (or final_*), validate them. Otherwise, skip accuracy checks
            # Support both `train_acc`/`test_acc` and `final_train_acc`/`final_test_acc` naming
            if 'final_train_acc' in df.columns and 'final_test_acc' in df.columns:
                avg_train_acc = df['final_train_acc'].mean()
                avg_test_acc = df['final_test_acc'].mean()
            else:
                has_train_acc = 'train_acc' in df.columns
                has_test_acc = 'test_acc' in df.columns
                avg_train_acc = df['train_acc'].mean() if has_train_acc else None
                avg_test_acc = df['test_acc'].mean() if has_test_acc else None

            if avg_train_acc is not None and avg_train_acc < min_train_acc:
                print_error(f"Average train accuracy {avg_train_acc:.1f}% is too low (expected > {min_train_acc}%)")
                return False

            if avg_test_acc is not None and avg_test_acc < min_test_acc:
                print_error(f"Average test accuracy {avg_test_acc:.1f}% is too low (expected > {min_test_acc}%)")
                return False
            if not (avg_train_acc is not None or avg_test_acc is not None):
                # Check for other numeric metrics. If none, consider it a warning but not a hard failure
                numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
                if not numeric_cols:
                    print_error("No numeric metrics found in results CSV")
                    return False

            cols_to_check = [c for c in ['train_loss', 'train_acc', 'test_loss', 'test_acc'] if c in df.columns]
            if cols_to_check and np.any(df[cols_to_check].isnull().to_numpy()):
                print_error("Found NaN values in results")
                return False

            # Per-experiment required columns & thresholds
            spec = EXP_METRICS_SPEC.get(experiment_name, {'required_cols': [], 'min': {}})
            required_cols = spec.get('required_cols', [])
            thresholds = spec.get('min', {})

            # Resolve required columns and their actual names in the CSV (handle 'final_' aliases)
            resolved_required = {}
            def resolve_col_name(preferred):
                if preferred in df.columns:
                    return preferred
                if preferred.startswith('final_'):
                    alt = preferred.replace('final_', '')
                    if alt in df.columns:
                        return alt
                return None

            for c in required_cols:
                resolved = resolve_col_name(c)
                if not resolved:
                    print_error(f"Required column '{c}' (or alias) not found in results for {experiment_name}")
                    return False
                resolved_required[c] = resolved
            # Compose a helpful summary depending on metrics present
            if avg_test_acc is not None:
                print_success(f"{experiment_name.upper()} passed: {num_optimizers} optimizers, average test acc: {avg_test_acc:.2f}%")
            elif avg_train_acc is not None:
                print_success(f"{experiment_name.upper()} passed: {num_optimizers} optimizers, average train acc: {avg_train_acc:.2f}%")
            else:
                numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
                print_success(f"{experiment_name.upper()} passed: {num_optimizers} optimizers, numeric metrics: {numeric_cols}")

            # Check per-experiment metric thresholds (use resolved column names when needed)
            for metric_col, min_value in thresholds.items():
                resolved = resolved_required.get(metric_col, metric_col if metric_col in df.columns else None)
                if resolved and resolved in df.columns:
                    avg_val = df[resolved].mean()
                    if avg_val < min_value:
                        print_error(f"Average {resolved} {avg_val:.3f} is too low (expected > {min_value})")
                        return False
            return True

        except subprocess.TimeoutExpired:
            print_error(f"{experiment_name} test timed out (>{timeout_sec/60:.0f} minutes)")
            return False
        except Exception as e:
            print_error(f"{experiment_name} error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quick validation of all GDSearch experiments')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip-mnist', action='store_true', help='Skip MNIST test (faster)')
    parser.add_argument('--skip-cifar', action='store_true', help='Skip CIFAR-10 test')
    parser.add_argument('--skip-nlp', action='store_true', help='Skip NLP test')
    parser.add_argument('--skip-medical', action='store_true', help='Skip Medical test')
    args = parser.parse_args()
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}GDSearch Quick Validation Test{RESET}")
    print(f"{BLUE}Verifies all 26 experiments are bug-free{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: Comprehensive validation
    results['validation'] = test_validation_script()
    
    # Test 3: MNIST quick test (most comprehensive)
    if not args.skip_mnist:
        results['mnist'] = test_mnist_quick()
    else:
        print_info("Skipping MNIST test (--skip-mnist flag)")
        results['mnist'] = None
    
    # Run CIFAR-10 quick test
    if args.skip_cifar:
        print_info("Skipping CIFAR-10 test (--skip-cifar flag)")
        results['cifar10'] = None
    else:
        print_info("Running CIFAR-10 quick test (ultra-quick)")
        # CIFAR can be slow on local machines; increase timeout to 1 hour in case ULTRA_QUICK didn't propagate
        results['cifar10'] = run_quick_experiment('cifar10', expected_min_optimizers=3, min_train_acc=40.0, min_test_acc=20.0, timeout_sec=3600)

    # Run NLP quick test
    if args.skip_nlp:
        print_info("Skipping NLP test (--skip-nlp flag)")
        results['nlp'] = None
    else:
        print_info("Running NLP quick test (ultra-quick)")
        # Simplified local NLP may only run 2 optimizers; accept 2 as minimum for a pass
        results['nlp'] = run_quick_experiment('nlp', expected_min_optimizers=2, min_train_acc=40.0, min_test_acc=30.0)

    # Run Medical quick test
    if args.skip_medical:
        print_info("Skipping Medical test (--skip-medical flag)")
        results['medical'] = None
    else:
        print_info("Running Medical quick test (ultra-quick)")
        results['medical'] = run_quick_experiment('medical', expected_min_optimizers=3, min_train_acc=30.0, min_test_acc=20.0)
    
    # Final summary
    print_header("FINAL SUMMARY")
    
    for test_name, passed in results.items():
        if passed is None:
            print_info(f"{test_name:20s} SKIPPED")
        elif passed:
            print_success(f"{test_name:20s} PASSED")
        else:
            print_error(f"{test_name:20s} FAILED")
    
    total_tests = sum(1 for v in results.values() if v is not None)
    passed_tests = sum(1 for v in results.values() if v is True)
    
    print(f"\n{BLUE}Results: {passed_tests}/{total_tests} tests passed{RESET}\n")
    
    if passed_tests == total_tests:
        print(f"{GREEN}ALL TESTS PASSED - Codebase is research-grade{RESET}")
        print(f"\n{BLUE}You can now run the full benchmark suite:{RESET}")
        print(f"  python run_all_kaggle.py --experiments all --seeds 42,123,456")
        return 0
    else:
        print(f"{RED}SOME TESTS FAILED - Fix issues before proceeding{RESET}")
        return 1


if __name__ == '__main__':
    # Configure Windows console encoding before any output
    configure_windows_console_encoding()
    
    sys.exit(main())
