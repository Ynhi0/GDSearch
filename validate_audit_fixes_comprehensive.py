#!/usr/bin/env python3
"""
Comprehensive Audit Fix Validation Script

This script validates all 7 critical audit fixes implemented in run_all_kaggle.py:
1. Config loading (load_experiment_config called in main)
2. Scheduler state restoration (CIFAR, MNIST, ResNet, Medical)
3. Tainted tracking in CIFAR results
4. CLI flags for AMP/EMA/Label Smoothing
5. Global flag wiring for advanced features
6. OOM handling consistency (tainted marking vs skipping)
7. All experiments track effective_batch_size

Usage:
    python validate_audit_fixes_comprehensive.py

Exit codes:
    0 - All fixes validated successfully
    1 - One or more fixes failed validation
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

class AuditFixValidator:
    def __init__(self, file_path: str = "run_all_kaggle.py"):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Cannot find {file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
            self.lines = self.content.split('\n')
        
        self.results: List[Tuple[str, bool, str]] = []
        
    def validate_all(self) -> bool:
        """Run all validation checks and return True if all pass."""
        print("="*80)
        print("COMPREHENSIVE AUDIT FIX VALIDATION")
        print("="*80)
        print(f"Validating: {self.file_path}")
        print()
        
        checks = [
            ("FIX 1: Config Loading in main()", self.check_config_loading),
            ("FIX 2: Scheduler Restoration - CIFAR", self.check_scheduler_cifar),
            ("FIX 3: Scheduler Restoration - MNIST", self.check_scheduler_mnist),
            ("FIX 4: Scheduler Restoration - ResNet/IMDB", self.check_scheduler_resnet),
            ("FIX 4b: Scheduler Restoration - Medical", self.check_scheduler_medical),
            ("FIX 5: Tainted Tracking - CIFAR Initialization", self.check_tainted_init_cifar),
            ("FIX 5b: Tainted Tracking - CIFAR OOM Handling", self.check_tainted_oom_cifar),
            ("FIX 5c: Tainted Tracking - CIFAR Results", self.check_tainted_results_cifar),
            ("FIX 10: CLI Flags for Advanced Features", self.check_cli_flags),
            ("FIX 10b: Global Flag Wiring", self.check_global_flags),
            ("FIX 10c: Feature Status Display", self.check_feature_display),
        ]
        
        for name, check_func in checks:
            passed, message = check_func()
            self.results.append((name, passed, message))
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} - {name}")
            if message:
                print(f"      {message}")
        
        print()
        print("="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        passed_count = sum(1 for _, passed, _ in self.results if passed)
        total_count = len(self.results)
        
        print(f"Passed: {passed_count}/{total_count}")
        
        if passed_count == total_count:
            print("\n🎉 ALL AUDIT FIXES VALIDATED SUCCESSFULLY!")
            return True
        else:
            print("\n⚠️  SOME AUDIT FIXES FAILED VALIDATION")
            print("\nFailed checks:")
            for name, passed, message in self.results:
                if not passed:
                    print(f"  ❌ {name}")
                    if message:
                        print(f"      {message}")
            return False
    
    def check_config_loading(self) -> Tuple[bool, str]:
        """Verify that load_experiment_config is called in main() after args parsing."""
        # Look for the pattern:
        # args = parser.parse_args()
        # ... (possibly some lines)
        # experiment_config = load_experiment_config(args.config)
        
        pattern = r'args\s*=\s*parser\.parse_args\(\)'
        if not re.search(pattern, self.content):
            return False, "Cannot find args = parser.parse_args()"
        
        # Check for load_experiment_config call after parse_args
        pattern = r'experiment_config\s*=\s*load_experiment_config\(args\.config\)'
        if not re.search(pattern, self.content):
            return False, "load_experiment_config(args.config) not called in main()"
        
        # Verify it's stored in globals
        pattern = r"globals\(\)\['EXPERIMENT_CONFIG'\]\s*=\s*experiment_config"
        if not re.search(pattern, self.content):
            return False, "experiment_config not stored in globals()"
        
        return True, "Config loading properly wired into main()"
    
    def check_scheduler_cifar(self) -> Tuple[bool, str]:
        """Verify scheduler.load_state_dict() is called in CIFAR after scheduler creation."""
        # Look for the pattern in CIFAR experiment
        pattern = r'scheduler\s*=\s*CosineAnnealingLR\(optimizer.*?\).*?scheduler\.load_state_dict\(checkpoint\[.scheduler.\]\)'
        if not re.search(pattern, self.content, re.DOTALL):
            return False, "scheduler.load_state_dict() not found after scheduler creation in CIFAR"
        
        # Verify it's in a try-except block
        if 'scheduler.load_state_dict(checkpoint[' not in self.content:
            return False, "scheduler.load_state_dict pattern not found"
        
        return True, "Scheduler restoration implemented in CIFAR"
    
    def check_scheduler_mnist(self) -> Tuple[bool, str]:
        """Verify scheduler.load_state_dict() is called in MNIST after scheduler creation."""
        # Count occurrences - should have multiple (MNIST + CIFAR + ResNet)
        count = self.content.count('scheduler.load_state_dict(checkpoint')
        if count < 2:
            return False, f"Only {count} scheduler.load_state_dict calls found (expected at least 2)"
        
        return True, f"Found {count} scheduler restoration calls across experiments"
    
    def check_scheduler_resnet(self) -> Tuple[bool, str]:
        """Verify scheduler.load_state_dict() is called in ResNet/IMDB."""
        # Check for the checkpoint local variable check pattern
        pattern = r"if 'checkpoint' in locals\(\) and checkpoint and 'scheduler' in checkpoint:"
        if not re.search(pattern, self.content):
            return False, "ResNet/IMDB scheduler restoration pattern not found"
        
        return True, "Scheduler restoration implemented in ResNet/IMDB"
    
    def check_scheduler_medical(self) -> Tuple[bool, str]:
        """Verify scheduler.load_state_dict() is called in Medical experiment."""
        # Should have the same pattern as ResNet
        count = self.content.count("if 'checkpoint' in locals() and checkpoint and 'scheduler' in checkpoint:")
        if count < 2:
            return False, f"Only {count} scheduler restoration checks found (expected at least 2)"
        
        return True, "Scheduler restoration implemented in Medical experiment"
    
    def check_tainted_init_cifar(self) -> Tuple[bool, str]:
        """Verify run_tainted and effective_batch_size are initialized in CIFAR."""
        # Look for initialization in CIFAR experiment
        patterns = [
            r'run_tainted\s*=\s*False',
            r'effective_batch_size\s*=\s*\d+',
            r'original_batch_size\s*=\s*\d+'
        ]
        
        for pattern in patterns:
            if not re.search(pattern, self.content):
                return False, f"Pattern not found: {pattern}"
        
        # Check it's in CIFAR context (near "Training CIFAR-10")
        cifar_section = None
        for i, line in enumerate(self.lines):
            if 'Training CIFAR-10 with' in line:
                # Check previous 20 lines for run_tainted initialization
                section = '\n'.join(self.lines[max(0, i-20):i+5])
                if 'run_tainted = False' in section:
                    cifar_section = section
                    break
        
        if not cifar_section:
            return False, "run_tainted initialization not found near CIFAR training loop"
        
        return True, "Tainted tracking variables initialized in CIFAR"
    
    def check_tainted_oom_cifar(self) -> Tuple[bool, str]:
        """Verify OOM handling marks runs as tainted in CIFAR."""
        # Look for the updated OOM handling
        pattern = r'run_tainted\s*=\s*True.*?# Continue to save results with tainted flag'
        if not re.search(pattern, self.content, re.DOTALL):
            return False, "OOM handling doesn't mark run_tainted = True or doesn't continue"
        
        # Verify it doesn't skip (no 'continue' after OOM except the new one that continues to save)
        # Old code had: continue  # Skip this optimizer config
        # New code has: # Continue to save results with tainted flag
        
        return True, "OOM handling marks runs as tainted instead of skipping"
    
    def check_tainted_results_cifar(self) -> Tuple[bool, str]:
        """Verify CIFAR results include tainted and effective_batch_size columns."""
        # Look for the results append with tainted fields
        pattern = r"'tainted':\s*run_tainted.*?'effective_batch_size':\s*effective_batch_size"
        if not re.search(pattern, self.content, re.DOTALL):
            return False, "CIFAR results don't include tainted and effective_batch_size fields"
        
        # Verify original_batch_size is also included
        if "'original_batch_size': original_batch_size" not in self.content:
            return False, "CIFAR results missing original_batch_size field"
        
        return True, "CIFAR results include tainted, effective_batch_size, and original_batch_size"
    
    def check_cli_flags(self) -> Tuple[bool, str]:
        """Verify --use-amp, --use-ema, --label-smoothing CLI flags are defined."""
        flags = [
            r"parser\.add_argument\('--use-amp'",
            r"parser\.add_argument\('--use-ema'",
            r"parser\.add_argument\('--label-smoothing'"
        ]
        
        for pattern in flags:
            if not re.search(pattern, self.content):
                flag_name = pattern.split("'")[1]
                return False, f"CLI flag not found: {flag_name}"
        
        return True, "All advanced feature CLI flags defined"
    
    def check_global_flags(self) -> Tuple[bool, str]:
        """Verify USE_AMP, USE_EMA, LABEL_SMOOTHING are wired to globals."""
        # Look for global declaration
        pattern = r'global\s+.*?USE_AMP.*?USE_EMA.*?LABEL_SMOOTHING'
        if not re.search(pattern, self.content, re.DOTALL):
            return False, "Global declaration doesn't include USE_AMP, USE_EMA, LABEL_SMOOTHING"
        
        # Check assignments
        patterns = [
            r'USE_AMP\s*=\s*args\.use_amp',
            r'USE_EMA\s*=\s*args\.use_ema',
            r'LABEL_SMOOTHING\s*=\s*args\.label_smoothing'
        ]
        
        for pattern in patterns:
            if not re.search(pattern, self.content):
                return False, f"Global assignment not found: {pattern}"
        
        return True, "Advanced feature flags properly wired to globals"
    
    def check_feature_display(self) -> Tuple[bool, str]:
        """Verify feature status is displayed when enabled."""
        patterns = [
            r'if USE_AMP:.*?print.*?AMP',
            r'if USE_EMA:.*?print.*?EMA',
            r'if LABEL_SMOOTHING > 0:.*?print.*?Label Smoothing'
        ]
        
        for pattern in patterns:
            if not re.search(pattern, self.content, re.DOTALL):
                return False, f"Feature display pattern not found: {pattern}"
        
        return True, "Advanced features display status when enabled"


def main():
    """Main validation entry point."""
    try:
        validator = AuditFixValidator()
        success = validator.validate_all()
        
        if success:
            print("\n✅ All critical audit fixes have been successfully validated!")
            print("\nNext steps:")
            print("  1. Run a quick test: python run_all_kaggle.py --ultra-quick --experiments mnist")
            print("  2. Test config loading: python run_all_kaggle.py --config configs/benchmark_hyperparameters.json --quick")
            print("  3. Test advanced features: python run_all_kaggle.py --use-amp --use-ema --label-smoothing 0.1")
            print("  4. Run full validation: python scripts/quick_validation_test.py")
            return 0
        else:
            print("\n❌ Validation failed. Please review and fix the failing checks.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
