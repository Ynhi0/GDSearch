"""
Verification script for missing infrastructure modules.

This script verifies that the following modules exist and are properly structured:
1. src/utils/csv_utils.py - CSV utilities with safety and error handling
2. src/utils/checkpoint_utils.py - Checkpoint management with atomic saves
3. src/utils/parallel_experiment_runner.py - Parallel experiment execution

Author: GDSearch Codebase Janitor
Date: 2026-02-03
"""
import sys
import inspect
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def verify_module_structure():
    """Verify that all infrastructure modules exist and are importable."""
    
    print("=" * 80)
    print("MODULE STRUCTURE VERIFICATION")
    print("=" * 80)
    
    # Check 1: csv_utils.py
    print("\n1. VERIFYING src/utils/csv_utils.py")
    print("-" * 80)
    try:
        from src.utils.csv_utils import safe_read_csv, cleanup_empty_csvs, CSVReadError
        
        print(f"✅ Module: {safe_read_csv.__module__}")
        print(f"✅ safe_read_csv signature: {inspect.signature(safe_read_csv)}")
        print(f"✅ cleanup_empty_csvs signature: {inspect.signature(cleanup_empty_csvs)}")
        print(f"✅ CSVReadError exception class defined")
        
        # Check docstring
        if safe_read_csv.__doc__:
            print(f"✅ Docstring present (length: {len(safe_read_csv.__doc__)} chars)")
        
        print("✅ csv_utils.py: ALL CHECKS PASSED")
        
    except ImportError as e:
        print(f"❌ FAILED: Cannot import csv_utils: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    # Check 2: checkpoint_utils.py
    print("\n2. VERIFYING src/utils/checkpoint_utils.py")
    print("-" * 80)
    try:
        from src.utils.checkpoint_utils import (
            CheckpointManager,
            create_checkpoint,
            load_checkpoint_safe,
            save_checkpoint_atomic
        )
        
        print(f"✅ Module: {CheckpointManager.__module__}")
        print(f"✅ CheckpointManager.__init__ signature: {inspect.signature(CheckpointManager.__init__)}")
        print(f"✅ create_checkpoint signature: {inspect.signature(create_checkpoint)}")
        print(f"✅ load_checkpoint_safe signature: {inspect.signature(load_checkpoint_safe)}")
        print(f"✅ save_checkpoint_atomic signature: {inspect.signature(save_checkpoint_atomic)}")
        
        # Check CheckpointManager methods
        required_methods = ['save_checkpoint', 'get_latest_checkpoint', 'get_best_checkpoint']
        for method in required_methods:
            if hasattr(CheckpointManager, method):
                print(f"✅ CheckpointManager.{method} exists")
            else:
                print(f"❌ CheckpointManager.{method} MISSING")
                return False
        
        print("✅ checkpoint_utils.py: ALL CHECKS PASSED")
        
    except ImportError as e:
        print(f"❌ FAILED: Cannot import checkpoint_utils: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    # Check 3: parallel_experiment_runner.py
    print("\n3. VERIFYING src/utils/parallel_experiment_runner.py")
    print("-" * 80)
    try:
        from src.utils.parallel_experiment_runner import (
            ParallelExperimentRunner,
            detect_gpu_configuration,
            run_experiment_on_gpu
        )
        
        print(f"✅ Module: {ParallelExperimentRunner.__module__}")
        print(f"✅ ParallelExperimentRunner.__init__ signature: {inspect.signature(ParallelExperimentRunner.__init__)}")
        print(f"✅ detect_gpu_configuration signature: {inspect.signature(detect_gpu_configuration)}")
        print(f"✅ run_experiment_on_gpu signature: {inspect.signature(run_experiment_on_gpu)}")
        
        # Check ParallelExperimentRunner methods
        required_methods = ['run_experiments_parallel', '_worker', '_run_sequential']
        for method in required_methods:
            if hasattr(ParallelExperimentRunner, method):
                print(f"✅ ParallelExperimentRunner.{method} exists")
            else:
                print(f"❌ ParallelExperimentRunner.{method} MISSING")
                return False
        
        # Test GPU configuration detection (should work even without GPUs)
        gpu_config = detect_gpu_configuration()
        print(f"✅ GPU detection works: {gpu_config['gpu_count']} GPUs detected")
        
        print("✅ parallel_experiment_runner.py: ALL CHECKS PASSED")
        
    except ImportError as e:
        print(f"❌ FAILED: Cannot import parallel_experiment_runner: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    # Check 4: Verify __init__.py exists
    print("\n4. VERIFYING src/utils/__init__.py")
    print("-" * 80)
    init_path = Path("src/utils/__init__.py")
    if init_path.exists():
        print(f"✅ {init_path} exists")
        print(f"✅ File size: {init_path.stat().st_size} bytes")
        
        # Try importing the package itself
        try:
            import src.utils
            print(f"✅ Package src.utils is importable")
            if hasattr(src.utils, '__all__'):
                print(f"✅ __all__ defined with {len(src.utils.__all__)} exports")
        except ImportError as e:
            print(f"⚠️  Warning: Cannot import src.utils package: {e}")
    else:
        print(f"❌ FAILED: {init_path} does not exist")
        return False
    
    # Check 5: Verify core modules
    print("\n5. VERIFYING src/core/tuning_cache.py")
    print("-" * 80)
    try:
        from src.core.tuning_cache import TuningCache, create_tuning_cache
        
        print(f"✅ Module: {TuningCache.__module__}")
        print(f"✅ TuningCache.__init__ signature: {inspect.signature(TuningCache.__init__)}")
        print(f"✅ create_tuning_cache signature: {inspect.signature(create_tuning_cache)}")
        
        # Check TuningCache methods
        required_methods = ['save_tuned_params', 'load_tuned_params', 'get_cache_key']
        for method in required_methods:
            if hasattr(TuningCache, method):
                print(f"✅ TuningCache.{method} exists")
            else:
                print(f"❌ TuningCache.{method} MISSING")
                return False
        
        print("✅ tuning_cache.py: ALL CHECKS PASSED")
        
    except ImportError as e:
        print(f"❌ FAILED: Cannot import tuning_cache: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    # Check 6: Verify resume_utils
    print("\n6. VERIFYING src/core/resume_utils.py")
    print("-" * 80)
    try:
        from src.core.resume_utils import compute_run_signature, decide_resume_action, results_exist
        
        print(f"✅ compute_run_signature signature: {inspect.signature(compute_run_signature)}")
        print(f"✅ decide_resume_action signature: {inspect.signature(decide_resume_action)}")
        print(f"✅ results_exist signature: {inspect.signature(results_exist)}")
        
        print("✅ resume_utils.py: ALL CHECKS PASSED")
        
    except ImportError as e:
        print(f"❌ FAILED: Cannot import resume_utils: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ ALL VERIFICATION CHECKS PASSED")
    print("=" * 80)
    print("\nSUMMARY:")
    print("  • csv_utils.py: 2 functions + 1 exception class")
    print("  • checkpoint_utils.py: 1 class + 3 functions")
    print("  • parallel_experiment_runner.py: 1 class + 2 functions")
    print("  • __init__.py: Package structure defined")
    print("  • tuning_cache.py: 1 class + 1 function")
    print("  • resume_utils.py: 3 functions")
    print("\nAll modules are importable and have the expected structure.")
    print("=" * 80)
    
    return True


def verify_usage_in_codebase():
    """Verify that these modules can be used as expected in the codebase."""
    
    print("\n" + "=" * 80)
    print("USAGE PATTERN VERIFICATION")
    print("=" * 80)
    
    # Test 1: csv_utils usage
    print("\n1. Testing csv_utils usage pattern")
    print("-" * 80)
    try:
        from src.utils.csv_utils import safe_read_csv
        from pathlib import Path
        
        # Create a test CSV
        test_csv = Path("test_verify.csv")
        test_csv.write_text("col1,col2\n1,2\n3,4\n")
        
        # Test reading
        df = safe_read_csv(test_csv)
        if df is not None and len(df) == 2:
            print("✅ safe_read_csv successfully read test CSV")
        else:
            print("❌ safe_read_csv failed to read test CSV correctly")
            return False
        
        # Cleanup
        test_csv.unlink()
        
    except Exception as e:
        print(f"❌ csv_utils usage test failed: {e}")
        return False
    
    # Test 2: checkpoint_utils usage
    print("\n2. Testing checkpoint_utils usage pattern")
    print("-" * 80)
    try:
        from src.utils.checkpoint_utils import CheckpointManager
        from pathlib import Path
        import tempfile
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                keep_last=3,
                keep_best=2
            )
            print("✅ CheckpointManager instantiated successfully")
            
            # Verify methods are callable
            latest = manager.get_latest_checkpoint()
            best = manager.get_best_checkpoint()
            print("✅ CheckpointManager methods are callable")
        
    except Exception as e:
        print(f"❌ checkpoint_utils usage test failed: {e}")
        return False
    
    # Test 3: parallel_experiment_runner usage
    print("\n3. Testing parallel_experiment_runner usage pattern")
    print("-" * 80)
    try:
        from src.utils.parallel_experiment_runner import ParallelExperimentRunner, detect_gpu_configuration
        from pathlib import Path
        import tempfile
        
        # Test GPU detection
        gpu_info = detect_gpu_configuration()
        print(f"✅ detect_gpu_configuration returned: {gpu_info['gpu_count']} GPUs")
        
        # Create runner (should work even without GPUs)
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ParallelExperimentRunner(
                num_gpus=0,  # Use 0 to test fallback
                results_dir=Path(tmpdir),
                strict=False
            )
            print("✅ ParallelExperimentRunner instantiated successfully")
        
    except Exception as e:
        print(f"❌ parallel_experiment_runner usage test failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL USAGE PATTERN TESTS PASSED")
    print("=" * 80)
    
    return True


def main():
    """Main verification function."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  INFRASTRUCTURE MODULES VERIFICATION REPORT".center(78) + "║")
    print("║" + "  GDSearch Codebase Janitor - Manual Quality Assurance".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run verification
    structure_ok = verify_module_structure()
    
    if not structure_ok:
        print("\n❌ MODULE STRUCTURE VERIFICATION FAILED")
        sys.exit(1)
    
    usage_ok = verify_usage_in_codebase()
    
    if not usage_ok:
        print("\n❌ USAGE PATTERN VERIFICATION FAILED")
        sys.exit(1)
    
    # Final report
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  ✅ ALL VERIFICATIONS PASSED".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  The following modules are now available and working:".center(78) + "║")
    print("║" + "    • src/utils/csv_utils.py".center(78) + "║")
    print("║" + "    • src/utils/checkpoint_utils.py".center(78) + "║")
    print("║" + "    • src/utils/parallel_experiment_runner.py".center(78) + "║")
    print("║" + "    • src/utils/__init__.py".center(78) + "║")
    print("║" + "    • src/core/tuning_cache.py".center(78) + "║")
    print("║" + "    • src/core/resume_utils.py".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")


if __name__ == "__main__":
    main()
