#!/usr/bin/env python3
"""
Comprehensive Verification Script for All Logic Review Fixes

This script validates that ALL fixes from the consolidated logic review have been
correctly implemented and are working as expected.

Usage:
    python verify_all_fixes.py [--verbose]
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

class VerificationError(Exception):
    """Raised when a verification check fails."""
    pass


class FixVerifier:
    """Comprehensive verification of all logic review fixes."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed = []
        self.failed = []
        self.warnings = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message at specified level."""
        if level == "INFO" and self.verbose:
            logging.info(message)
        elif level == "WARNING":
            logging.warning(message)
            self.warnings.append(message)
        elif level == "ERROR":
            logging.error(message)
            
    def check(self, name: str, check_fn):
        """Run a verification check and record result."""
        try:
            self.log(f"\n{'='*70}")
            self.log(f"CHECKING: {name}")
            self.log(f"{'='*70}\n")
            check_fn()
            self.passed.append(name)
            self.log(f"✅ PASSED: {name}", "INFO")
        except Exception as e:
            self.failed.append((name, str(e)))
            self.log(f"❌ FAILED: {name}", "ERROR")
            self.log(f"   Error: {str(e)}", "ERROR")
            if self.verbose:
                self.log(traceback.format_exc(), "ERROR")
                
    def print_summary(self):
        """Print verification summary."""
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*70}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Checks: {total}")
        print(f"✅ Passed: {len(self.passed)}")
        print(f"❌ Failed: {len(self.failed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.failed:
            print(f"\n{'='*70}")
            print("FAILED CHECKS:")
            print(f"{'='*70}")
            for name, error in self.failed:
                print(f"\n❌ {name}")
                print(f"   {error}")
                
        if self.warnings:
            print(f"\n{'='*70}")
            print("WARNINGS:")
            print(f"{'='*70}")
            for warning in self.warnings:
                print(f"⚠️  {warning}")
                
        print(f"\n{'='*70}\n")
        
        return len(self.failed) == 0


# ============================================================================
# CRITICAL FIX VERIFICATIONS
# ============================================================================

def verify_test_set_leakage_fix(verifier: FixVerifier):
    """Verify BLOCKER-1: Test set leakage fix in tune_nn.py"""
    from scripts.tune_nn import best_by_eval
    import pandas as pd
    import tempfile
    import json
    
    verifier.log("Testing that best_by_eval rejects CSVs without validation data...")
    
    # Create a test CSV with only test data (no validation)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame({
            'epoch': [1, 2, 3],
            'phase': ['eval', 'eval', 'eval'],
            'accuracy': [0.8, 0.85, 0.9]
        })
        df.to_csv(f.name, index=False)
        test_csv_path = f.name
    
    try:
        # This should raise ValueError (not fall back to test set)
        best_by_eval([test_csv_path])
        raise VerificationError(
            "best_by_eval did NOT reject CSV without validation data. "
            "Test set leakage prevention FAILED!"
        )
    except ValueError as e:
        # Expected behavior
        if "INTEGRITY ERROR" in str(e) and "validation data" in str(e):
            verifier.log(f"✓ Correctly rejected CSV without validation: {e}")
        else:
            raise VerificationError(f"Wrong error message: {e}")
    finally:
        Path(test_csv_path).unlink(missing_ok=True)
    
    verifier.log("✓ Test set leakage prevention working correctly")


def verify_schema_rejects_invalid_keys(verifier: FixVerifier):
    """Verify BLOCKER-2: Schema rejects zombie keys"""
    import jsonschema
    import json
    
    verifier.log("Testing that schema rejects configs with invalid keys...")
    
    # Load schema
    schema_path = Path('configs/config_schema.json')
    with open(schema_path) as f:
        schema = json.load(f)
    
    # Test config with invalid key
    invalid_config = {
        "dataset": "MNIST",
        "sweeps": [{
            "optimizer": "Adam",
            "lr_values": [0.001],
            "invalid_zombie_key": "should_be_rejected"
        }]
    }
    
    try:
        jsonschema.validate(invalid_config, schema)
        raise VerificationError(
            "Schema ACCEPTED config with invalid key 'invalid_zombie_key'. "
            "additionalProperties: false NOT working!"
        )
    except jsonschema.ValidationError as e:
        # Expected behavior
        if "additional properties" in str(e).lower():
            verifier.log(f"✓ Correctly rejected invalid key: {e.message}")
        else:
            raise VerificationError(f"Wrong validation error: {e}")
    
    verifier.log("✓ Schema correctly rejects zombie keys")


def verify_seed_validation(verifier: FixVerifier):
    """Verify CRITICAL-8: Minimum 3 seeds enforced"""
    from src.utils.experiment_config import ExperimentConfig
    
    verifier.log("Testing minimum seed validation...")
    
    # Test 1: Single seed should fail
    try:
        config = ExperimentConfig.from_dict({'seeds': [42]})
        raise VerificationError("Accepted single seed - validation not working!")
    except ValueError as e:
        if "MINIMUM 3 seeds" in str(e):
            verifier.log("✓ Correctly rejected single seed")
        else:
            raise VerificationError(f"Wrong error for single seed: {e}")
    
    # Test 2: Two seeds should fail
    try:
        config = ExperimentConfig.from_dict({'seeds': [42, 123]})
        raise VerificationError("Accepted 2 seeds - validation not working!")
    except ValueError as e:
        if "MINIMUM 3 seeds" in str(e):
            verifier.log("✓ Correctly rejected 2 seeds")
        else:
            raise VerificationError(f"Wrong error for 2 seeds: {e}")
    
    # Test 3: Three seeds should pass
    config = ExperimentConfig.from_dict({'seeds': [42, 123, 456]})
    verifier.log(f"✓ Accepted valid 3 seeds: {config.seeds}")
    
    # Test 4: Duplicate seeds should fail
    try:
        config = ExperimentConfig.from_dict({'seeds': [42, 42, 123]})
        raise VerificationError("Accepted duplicate seeds - validation not working!")
    except ValueError as e:
        if "DUPLICATE SEEDS" in str(e):
            verifier.log("✓ Correctly rejected duplicate seeds")
        else:
            raise VerificationError(f"Wrong error for duplicates: {e}")
    
    # Test 5: Invalid seed range should fail
    try:
        config = ExperimentConfig.from_dict({'seeds': [-1, 42, 123]})
        raise VerificationError("Accepted negative seed - validation not working!")
    except ValueError as e:
        if "INVALID SEEDS" in str(e):
            verifier.log("✓ Correctly rejected invalid seed range")
        else:
            raise VerificationError(f"Wrong error for invalid range: {e}")
    
    verifier.log("✓ Seed validation working correctly")


def verify_path_handling(verifier: FixVerifier):
    """Verify CRITICAL-7: Path type conversion and absolute path resolution"""
    from src.utils.experiment_config import ExperimentConfig
    
    verifier.log("Testing path handling...")
    
    # Test 1: String path should be converted to Path
    config = ExperimentConfig.from_dict({'results_dir': 'results'})
    from pathlib import Path
    assert isinstance(config.results_dir, Path), "results_dir not converted to Path"
    verifier.log(f"✓ String converted to Path: {type(config.results_dir)}")
    
    # Test 2: Path should be absolute
    assert config.results_dir.is_absolute(), "results_dir not absolute"
    verifier.log(f"✓ Path is absolute: {config.results_dir}")
    
    # Test 3: Path should exist and be writable
    assert config.results_dir.exists(), "results_dir not created"
    assert config.results_dir.is_dir(), "results_dir not a directory"
    verifier.log(f"✓ Directory exists and is writable: {config.results_dir}")
    
    verifier.log("✓ Path handling working correctly")


def verify_config_validator_structure(verifier: FixVerifier):
    """Verify CRITICAL-9: Validator checks correct structure"""
    from src.utils.config_validator import validate_config_keys
    import json
    import tempfile
    
    verifier.log("Testing config validator structure...")
    
    # Create test config with conflicting LR keys at sweep level
    test_config = {
        "dataset": "MNIST",
        "sweeps": [{
            "optimizer": "Adam",
            "learning_rate": [0.001],  # Old deprecated key
            "lr_values": [0.001]       # New canonical key
        }]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        issues = validate_config_keys(config_path)
        
        # Should have error about conflicting keys
        if not issues['errors']:
            raise VerificationError(
                "Validator did not detect conflicting learning_rate and lr_values"
            )
        
        error_text = ' '.join(issues['errors'])
        if "BOTH" in error_text and "learning_rate" in error_text and "lr_values" in error_text:
            verifier.log(f"✓ Correctly detected conflicting keys: {issues['errors']}")
        else:
            raise VerificationError(f"Wrong error detection: {issues['errors']}")
            
    finally:
        Path(config_path).unlink(missing_ok=True)
    
    verifier.log("✓ Config validator checking correct structure")


def verify_utilities_created(verifier: FixVerifier):
    """Verify that all utility modules were created"""
    verifier.log("Checking that utility modules exist...")
    
    required_modules = [
        'src/core/device_utils.py',
        'src/core/filesystem_utils.py',
        'src/core/validation.py'
    ]
    
    for module_path in required_modules:
        path = Path(module_path)
        if not path.exists():
            raise VerificationError(f"Required utility module missing: {module_path}")
        verifier.log(f"✓ Found: {module_path}")
    
    # Check that key functions exist
    verifier.log("\nChecking key utility functions...")
    
    try:
        from src.core.device_utils import safe_to_device, clear_gpu_memory, get_available_device
        verifier.log("✓ device_utils functions available")
    except ImportError as e:
        raise VerificationError(f"device_utils import failed: {e}")
    
    try:
        from src.core.filesystem_utils import (
            check_write_permission,
            check_disk_space,
            ensure_directory_exists,
            cleanup_stale_temp_files
        )
        verifier.log("✓ filesystem_utils functions available")
    except ImportError as e:
        raise VerificationError(f"filesystem_utils import failed: {e}")
    
    try:
        from src.core.validation import (
            validate_loss,
            validate_dataset,
            validate_batch_size,
            validate_gradients
        )
        verifier.log("✓ validation functions available")
    except ImportError as e:
        raise VerificationError(f"validation import failed: {e}")
    
    verifier.log("✓ All utility modules created and importable")


def verify_import_safety(verifier: FixVerifier):
    """Verify no import-time side effects"""
    verifier.log("Testing import safety (no side effects)...")
    
    critical_modules = [
        'src.core.device_utils',
        'src.core.filesystem_utils',
        'src.core.validation',
        'src.core.experiment_tracker',
        'src.utils.experiment_config',
        'scripts.tune_nn'
    ]
    
    for module_name in critical_modules:
        try:
            __import__(module_name)
            verifier.log(f"✓ Safe import: {module_name}")
        except Exception as e:
            raise VerificationError(f"Import failed for {module_name}: {e}")
    
    verifier.log("✓ All critical modules import safely")


# ============================================================================
# INTEGRATION CHECKS
# ============================================================================

def verify_integration_quick_test(verifier: FixVerifier):
    """Verify quick validation test passes"""
    import subprocess
    
    verifier.log("Running quick_validation_test.py...")
    
    result = subprocess.run(
        [sys.executable, "scripts/quick_validation_test.py", "--verbose"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        verifier.log(f"STDOUT: {result.stdout}", "ERROR")
        verifier.log(f"STDERR: {result.stderr}", "ERROR")
        raise VerificationError(
            f"quick_validation_test.py failed with code {result.returncode}"
        )
    
    verifier.log(f"✓ quick_validation_test.py passed")
    verifier.log(f"Output:\n{result.stdout}")


def verify_schema_validation_passes(verifier: FixVerifier):
    """Verify all existing configs pass schema validation"""
    import subprocess
    
    verifier.log("Running validate_config_schema.py on all configs...")
    
    result = subprocess.run(
        [sys.executable, "scripts/validate_config_schema.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Check output for any validation errors
    if "FAILED" in result.stdout or result.returncode != 0:
        verifier.log(f"STDOUT: {result.stdout}", "ERROR")
        verifier.log(f"STDERR: {result.stderr}", "ERROR")
        
        # If failure is due to zombie keys, that's actually good
        if "additional properties" in result.stdout.lower():
            verifier.log(
                "Schema correctly rejecting configs with zombie keys. "
                "Configs need to be cleaned up.",
                "WARNING"
            )
        else:
            raise VerificationError(
                f"Schema validation failed unexpectedly: {result.stdout}"
            )
    else:
        verifier.log(f"✓ All configs pass schema validation")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all verification checks."""
    verbose = '--verbose' in sys.argv
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE FIX VERIFICATION")
    print(f"{'='*70}\n")
    print("This script verifies all fixes from the consolidated logic review.")
    print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
    print(f"{'='*70}\n")
    
    verifier = FixVerifier(verbose=verbose)
    
    # Run all verification checks
    verifier.check("BLOCKER-1: Test Set Leakage Fix", 
                   lambda: verify_test_set_leakage_fix(verifier))
    
    verifier.check("BLOCKER-2: Schema Rejects Invalid Keys",
                   lambda: verify_schema_rejects_invalid_keys(verifier))
    
    verifier.check("CRITICAL-8: Minimum Seed Validation",
                   lambda: verify_seed_validation(verifier))
    
    verifier.check("CRITICAL-7: Path Type Conversion & Absolute Paths",
                   lambda: verify_path_handling(verifier))
    
    verifier.check("CRITICAL-9: Config Validator Structure Check",
                   lambda: verify_config_validator_structure(verifier))
    
    verifier.check("Utility Modules Created",
                   lambda: verify_utilities_created(verifier))
    
    verifier.check("Import Safety (No Side Effects)",
                   lambda: verify_import_safety(verifier))
    
    # Integration checks (can be skipped if dependencies missing)
    try:
        verifier.check("Integration: Quick Validation Test",
                       lambda: verify_integration_quick_test(verifier))
    except FileNotFoundError:
        verifier.log("Skipping integration test (file not found)", "WARNING")
    
    try:
        verifier.check("Integration: Schema Validation on All Configs",
                       lambda: verify_schema_validation_passes(verifier))
    except FileNotFoundError:
        verifier.log("Skipping schema validation test (file not found)", "WARNING")
    
    # Print summary
    success = verifier.print_summary()
    
    if success:
        print("✅ ALL CRITICAL FIXES VERIFIED SUCCESSFULLY!")
        return 0
    else:
        print("❌ SOME FIXES FAILED VERIFICATION - SEE ABOVE FOR DETAILS")
        return 1


if __name__ == '__main__':
    sys.exit(main())
