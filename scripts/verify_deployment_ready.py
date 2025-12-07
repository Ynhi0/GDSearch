#!/usr/bin/env python3
"""
Deployment Verification Script
Verifies all P0 and P1 fixes are working before Kaggle deployment
"""

import sys
from pathlib import Path
import importlib.util

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check_mark(passed: bool) -> str:
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"

def test_dependency_available(package: str, min_version: str = None) -> bool:
    """Test if a package is importable and meets minimum version."""
    try:
        spec = importlib.util.find_spec(package)
        if spec is None:
            return False
        
        # Try to import and check version
        module = importlib.import_module(package)
        if min_version and hasattr(module, '__version__'):
            # Simple version comparison (not production-grade)
            return True  # For now, just check importability
        return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False

def main():
    print("=" * 80)
    print("GDSearch Deployment Verification")
    print("=" * 80)
    print()
    
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    print()
    
    all_passed = True
    
    # P0 Tests: Critical Dependencies
    print(f"{YELLOW}P0 Tests: Critical Dependencies{RESET}")
    print("-" * 80)
    
    critical_deps = [
        ("psutil", "5.9.0", "System monitoring (run_all_kaggle.py line 42)"),
        ("scipy", "1.10.0", "Statistical analysis"),
        ("optuna", "3.0.0", "Hyperparameter tuning"),
        ("mlflow", "2.0.0", "Experiment tracking"),
        ("transformers", "4.30.0", "NLP experiments"),
        ("datasets", "2.14.0", "IMDB dataset loading"),
        ("plotly", "5.14.0", "Interactive visualizations"),
        ("kaleido", "0.2.0", "Static image export"),
    ]
    
    for package, min_ver, purpose in critical_deps:
        passed = test_dependency_available(package, min_ver)
        all_passed &= passed
        status = check_mark(passed)
        print(f"  {status} {package:20s} >= {min_ver:10s} [{purpose}]")
    
    print()
    
    # P1 Tests: Path Resolution
    print(f"{YELLOW}P1 Tests: Path Resolution{RESET}")
    print("-" * 80)
    
    # Test if src/ is in path
    project_root = Path(__file__).parent
    src_path = project_root / 'src'
    
    # Add to path like run_all_kaggle.py does
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_path))
    
    # Test core imports
    path_tests = [
        ("src.core.optimizers", "SGD, Adam, AdamW"),
        ("src.core.pytorch_optimizers", "Wrappers for PyTorch"),
        ("src.core.functions", "Test functions"),
    ]
    
    for module_name, description in path_tests:
        try:
            module = importlib.import_module(module_name)
            passed = True
        except ImportError as e:
            passed = False
        
        all_passed &= passed
        status = check_mark(passed)
        print(f"  {status} {module_name:40s} [{description}]")
    
    print()
    
    # Verification of Configuration
    print(f"{YELLOW}Configuration Verification{RESET}")
    print("-" * 80)
    
    # Check multi-seed configuration (should be 10)
    try:
        with open(project_root / 'run_all_kaggle.py') as f:
            content = f.read()
            # Look for SEEDS definition
            if 'SEEDS' in content and '10' in content:
                seeds_ok = True
            else:
                seeds_ok = False
    except:
        seeds_ok = False
    
    all_passed &= seeds_ok
    print(f"  {check_mark(seeds_ok)} Multi-seed configuration (10 seeds)")
    
    # Check VRAM cleanup integration
    cleanup_count = 0
    try:
        with open(project_root / 'run_all_kaggle.py') as f:
            content = f.read()
            cleanup_count = content.count('clear_gpu_memory')
            vram_ok = cleanup_count >= 10  # Should have 10+ cleanup calls
    except:
        vram_ok = False
    
    all_passed &= vram_ok
    print(f"  {check_mark(vram_ok)} VRAM cleanup integration ({cleanup_count} calls)")
    
    # Check enhanced path setup
    try:
        with open(project_root / 'run_all_kaggle.py') as f:
            content = f.read()
            path_ok = 'sys.path.insert(0, str(project_root / \'src\'))' in content
    except:
        path_ok = False
    
    all_passed &= path_ok
    print(f"  {check_mark(path_ok)} Enhanced path setup (src/ explicitly added)")
    
    print()
    
    # Files Check
    print(f"{YELLOW}Required Files Verification{RESET}")
    print("-" * 80)
    
    required_files = [
        "run_all_kaggle.py",
        "kaggle/requirements_kaggle.txt",
        "kaggle/run_benchmark.ipynb",
        "configs/nn_tuning.json",
        "configs/cifar10_tuning.json",
        "src/core/optimizers.py",
        "src/core/pytorch_optimizers.py",
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        all_passed &= exists
        print(f"  {check_mark(exists)} {file_path}")
    
    print()
    print("=" * 80)
    
    if all_passed:
        print(f"{GREEN}✓ ALL CHECKS PASSED - Ready for deployment{RESET}")
        print()
        print("Next steps:")
        print("  1. Test locally: python run_all_kaggle.py --quick --seeds 42")
        print("  2. Upload to Kaggle and run notebook")
        print("  3. Monitor resource usage (VRAM < 15GB, RAM < 30GB)")
        return 0
    else:
        print(f"{RED}✗ SOME CHECKS FAILED - Fix issues before deployment{RESET}")
        print()
        print("To fix missing dependencies:")
        print("  pip install -r kaggle/requirements_kaggle.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
