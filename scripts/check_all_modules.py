#!/usr/bin/env python3
"""
Quick module existence and import check for GDSearch.

This script verifies that all required modules exist and can be imported.
"""
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def check_module_file(module_path_str, description=""):
    """Check if a module file exists and return formatted status."""
    module_path = project_root / module_path_str
    
    if module_path.exists():
        size = module_path.stat().st_size
        size_str = format_size(size)
        print(f"   ✅ {module_path_str} ({size_str})")
        return True
    else:
        print(f"   ❌ MISSING: {module_path_str}")
        return False


def check_imports():
    """Test importing all critical modules."""
    print("\n" + "="*80)
    print("IMPORT VERIFICATION")
    print("="*80)
    
    errors = []
    
    # Core modules
    print("\n📦 Core modules (src/core/):")
    try:
        from src.core.tuning_cache import create_tuning_cache, TuningCache
        print("   ✅ tuning_cache: TuningCache, create_tuning_cache")
    except ImportError as e:
        print(f"   ❌ tuning_cache import failed: {e}")
        errors.append(("tuning_cache", e))
    
    try:
        from src.core.resume_utils import compute_run_signature, decide_resume_action, results_exist
        print("   ✅ resume_utils: compute_run_signature, decide_resume_action, results_exist")
    except ImportError as e:
        print(f"   ❌ resume_utils import failed: {e}")
        errors.append(("resume_utils", e))
    
    # Utility modules
    print("\n🔧 Utility modules (src/utils/):")
    try:
        from src.utils.csv_utils import safe_read_csv, cleanup_empty_csvs, CSVReadError
        print("   ✅ csv_utils: safe_read_csv, cleanup_empty_csvs, CSVReadError")
    except ImportError as e:
        print(f"   ❌ csv_utils import failed: {e}")
        errors.append(("csv_utils", e))
    
    try:
        from src.utils.checkpoint_utils import CheckpointManager, create_checkpoint, load_checkpoint_safe
        print("   ✅ checkpoint_utils: CheckpointManager, create_checkpoint, load_checkpoint_safe")
    except ImportError as e:
        print(f"   ❌ checkpoint_utils import failed: {e}")
        errors.append(("checkpoint_utils", e))
    
    try:
        from src.utils.parallel_experiment_runner import ParallelExperimentRunner, detect_gpu_configuration
        print("   ✅ parallel_experiment_runner: ParallelExperimentRunner, detect_gpu_configuration")
    except ImportError as e:
        print(f"   ❌ parallel_experiment_runner import failed: {e}")
        errors.append(("parallel_experiment_runner", e))
    
    try:
        from src.utils.constants import OptimizerNames, MNIST_MEAN, CIFAR10_MEAN
        print("   ✅ constants: OptimizerNames, MNIST_MEAN, CIFAR10_MEAN")
    except ImportError as e:
        print(f"   ❌ constants import failed: {e}")
        errors.append(("constants", e))
    
    try:
        from src.utils.device_safety import safe_device_transfer, gpu_safe_operation
        print("   ✅ device_safety: safe_device_transfer, gpu_safe_operation")
    except ImportError as e:
        print(f"   ❌ device_safety import failed: {e}")
        errors.append(("device_safety", e))
    
    return errors


def main():
    """Main check function."""
    print("\n" + "="*80)
    print("GDSearch Module Existence & Import Check")
    print("="*80)
    
    # File existence check
    print("\nCore modules (src/core/):")
    core_files = []
    core_files.append(check_module_file("src/core/__init__.py"))
    core_files.append(check_module_file("src/core/tuning_cache.py"))
    core_files.append(check_module_file("src/core/models.py"))
    core_files.append(check_module_file("src/core/optimizers.py"))
    core_files.append(check_module_file("src/core/checkpoint_manager.py"))
    core_files.append(check_module_file("src/core/experiment_tracker.py"))
    core_files.append(check_module_file("src/core/resume_utils.py"))
    core_files.append(check_module_file("src/core/training_utils.py"))
    
    print("\nUtility modules (src/utils/):")
    util_files = []
    util_files.append(check_module_file("src/utils/__init__.py"))
    util_files.append(check_module_file("src/utils/csv_utils.py"))
    util_files.append(check_module_file("src/utils/checkpoint_utils.py"))
    util_files.append(check_module_file("src/utils/parallel_experiment_runner.py"))
    util_files.append(check_module_file("src/utils/constants.py"))
    util_files.append(check_module_file("src/utils/device_safety.py"))
    
    # Import verification
    import_errors = check_imports()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    core_ok = all(core_files)
    util_ok = all(util_files)
    imports_ok = len(import_errors) == 0
    
    print(f"\n📁 File Existence:")
    print(f"   Core modules: {'✅ PASS' if core_ok else '❌ FAIL'} ({sum(core_files)}/{len(core_files)} files)")
    print(f"   Util modules: {'✅ PASS' if util_ok else '❌ FAIL'} ({sum(util_files)}/{len(util_files)} files)")
    
    print(f"\n📦 Import Tests:")
    if imports_ok:
        print(f"   ✅ PASS - All modules imported successfully")
    else:
        print(f"   ❌ FAIL - {len(import_errors)} import errors")
        for module_name, error in import_errors:
            print(f"      • {module_name}: {error}")
    
    print("\n" + "="*80)
    if core_ok and util_ok and imports_ok:
        print("✅ ALL CHECKS PASSED - All modules are present and working!")
        print("="*80)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - See errors above")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
