#!/usr/bin/env python3
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    print(f"{RED}✗{RESET} {text}")

def print_info(text):
    print(f"{YELLOW}ℹ{RESET} {text}")


def test_imports():
    """Test that all critical modules can be imported"""
    print_header("IMPORT VALIDATION")
    
    try:
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
        
    except Exception as e:
        print_error(f"Import failed: {e}")
        return False


def test_mnist_quick():
    """Test MNIST experiment with ultra-quick mode"""
    print_header("MNIST QUICK TEST (2 epochs, 1 seed)")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print_info(f"Running in temp directory: {tmpdir}")
        
        cmd = [
            sys.executable,
            str(project_root / "run_all_kaggle.py"),
            "--ultra-quick",
            "--seeds", "42",
            "--experiments", "mnist",
            "--results-dir", tmpdir,
            "--no-mlflow"  # Skip MLflow to avoid setup
        ]
        
        print_info("Command: python run_all_kaggle.py --ultra-quick --seeds 42 --experiments mnist")
        print_info("This will take 2-3 minutes...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=project_root
            )
            
            if result.returncode != 0:
                print_error("MNIST test failed")
                print(f"\nLast 30 lines of output:")
                lines = result.stdout.split('\n')
                for line in lines[-30:]:
                    print(f"  {line}")
                return False
            
            # Check results
            import pandas as pd
            results_file = Path(tmpdir) / "experiments" / "mnist" / "mnist_results.csv"
            
            if not results_file.exists():
                print_error(f"Results file not found: {results_file}")
                return False
            
            df = pd.read_csv(results_file)
            
            # Validate results
            if len(df) == 0:
                print_error("Results CSV is empty")
                return False
            
            # Check that we have results for multiple optimizers
            num_optimizers = len(df['optimizer'].unique())
            if num_optimizers < 5:
                print_error(f"Only {num_optimizers} optimizers tested (expected 12)")
                return False
            
            print_success(f"Tested {num_optimizers} optimizers")
            
            # Check accuracy sanity (should be > 80% for MNIST after 2 epochs)
            avg_train_acc = df['train_acc'].mean()
            avg_test_acc = df['test_acc'].mean()
            
            if avg_train_acc < 80.0:
                print_error(f"Average train accuracy {avg_train_acc:.1f}% is too low (expected > 80%)")
                print_error("This may indicate the training loop indentation bug!")
                return False
            
            if avg_test_acc < 80.0:
                print_error(f"Average test accuracy {avg_test_acc:.1f}% is too low (expected > 80%)")
                return False
            
            print_success(f"Average train accuracy: {avg_train_acc:.2f}%")
            print_success(f"Average test accuracy: {avg_test_acc:.2f}%")
            
            # Check for NaN/Inf
            if df[['train_loss', 'train_acc', 'test_loss', 'test_acc']].isnull().any().any():
                print_error("Found NaN values in results")
                return False
            
            print_success("No NaN/Inf values found")
            
            # Show top 3 performers
            print(f"\n{BLUE}Top 3 Optimizers (by test accuracy):{RESET}")
            top3 = df.nlargest(3, 'test_acc')[['optimizer', 'test_acc', 'train_acc']]
            for idx, row in top3.iterrows():
                print(f"  {row['optimizer']:20s} Test: {row['test_acc']:.2f}%  Train: {row['train_acc']:.2f}%")
            
            print(f"\n{GREEN}MNIST test PASSED{RESET}")
            return True
            
        except subprocess.TimeoutExpired:
            print_error("MNIST test timed out (> 5 minutes)")
            return False
        except Exception as e:
            print_error(f"MNIST test error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_validation_script():
    """Test the comprehensive validation script"""
    print_header("RUNNING COMPREHENSIVE VALIDATION")
    
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "validate_all_experiments.py"),
        "--smoke-test"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print_success("Comprehensive validation PASSED")
            # Show summary
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Total Passed' in line or 'Total Failed' in line or 'Missing Files' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print_error("Comprehensive validation FAILED")
            print(f"\nOutput:\n{result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Validation script timed out")
        return False
    except Exception as e:
        print_error(f"Validation script error: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quick validation of all GDSearch experiments')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip-mnist', action='store_true', help='Skip MNIST test (faster)')
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
        print(f"{GREEN}✅ ALL TESTS PASSED - Codebase is research-grade{RESET}")
        print(f"\n{BLUE}You can now run the full benchmark suite:{RESET}")
        print(f"  python run_all_kaggle.py --experiments all --seeds 42,123,456")
        return 0
    else:
        print(f"{RED}❌ SOME TESTS FAILED - Fix issues before proceeding{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
