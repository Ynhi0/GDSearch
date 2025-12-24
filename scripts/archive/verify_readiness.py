#!/usr/bin/env python3
"""
Readiness Verification Script

This script performs comprehensive end-to-end checks to verify the codebase
is ready for deployment and public release.

Addresses Phase 7 recommendation from the review report.
"""

import os
import sys
import subprocess
import logging
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ReadinessChecker:
    """Comprehensive readiness checker."""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        
    def log_pass(self, check_name):
        """Log a passed check."""
        self.checks_passed.append(check_name)
        logging.info(f"PASS: {check_name}")
        
    def log_fail(self, check_name, reason):
        """Log a failed check."""
        self.checks_failed.append((check_name, reason))
        logging.error(f"FAIL: {check_name}: {reason}")
        
    def log_warning(self, check_name, reason):
        """Log a warning."""
        self.warnings.append((check_name, reason))
        logging.warning(f"WARNING: {check_name}: {reason}")
    
    def check_requirements_file(self):
        """Check requirements.txt exists and is valid."""
        logging.info("\n" + "="*80)
        logging.info("CHECK 1: Requirements File")
        logging.info("="*80)
        
        req_path = Path('requirements.txt')
        if not req_path.exists():
            self.log_fail("requirements.txt", "File not found")
            return False
        
        self.log_pass("requirements.txt exists")
        
        # Check for critical packages
        with open(req_path, encoding='utf-8') as f:
            content = f.read()
        
        critical_packages = ['torch', 'numpy', 'pandas', 'matplotlib', 'scipy']
        for pkg in critical_packages:
            if pkg in content.lower():
                self.log_pass(f"  - {pkg} listed")
            else:
                self.log_warning(f"  - {pkg}", "Not found in requirements")
        
        # Check for conflict resolution
        if 'datasets>=2.14.0,<3.0.0' in content and 'pyarrow>=14.0.0,<20.0.0' in content:
            self.log_pass("  - Dependency conflict RESOLVED (datasets/pyarrow)")
        else:
            self.log_warning("  - Dependency versions", "Kaggle conflict may exist")
        
        return True
    
    def check_gpu_availability(self):
        """Check GPU availability."""
        logging.info("\n" + "="*80)
        logging.info("CHECK 2: GPU Availability")
        logging.info("="*80)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            self.log_pass(f"GPU available: {gpu_name}")
            self.log_pass(f"  - GPU count: {gpu_count}")
            self.log_pass(f"  - VRAM: {memory_gb:.2f} GB")
            
            if memory_gb < 4:
                self.log_warning("  - Low VRAM", "May need --adaptive-batch flag")
            
            return True
        else:
            self.log_warning("GPU", "No GPU available - will run on CPU (slow)")
            return False
    
    def check_results_directory(self):
        """Check results directory is writable."""
        logging.info("\n" + "="*80)
        logging.info("CHECK 3: Results Directory")
        logging.info("="*80)
        
        results_dir = Path('results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write
        test_file = results_dir / '.write_test'
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            test_file.unlink()
            self.log_pass("results/ directory writable")
            return True
        except Exception as e:
            self.log_fail("results/ directory", f"Not writable: {e}")
            return False
    
    def check_src_imports(self):
        """Check critical src modules can be imported."""
        logging.info("\n" + "="*80)
        logging.info("CHECK 4: Core Module Imports")
        logging.info("="*80)
        
        sys.path.insert(0, str(Path.cwd()))
        
        critical_modules = [
            ('src.core.optimizers', 'SGDOptimizer'),
            ('src.core.models', 'SimpleMLP'),
            ('src.core.training_enhancements', 'LRFinder'),
            ('src.core.training_enhancements', 'MemoryAwareBatchSizer'),
            ('src.experiments.beta_sensitivity_training', 'run_momentum_beta_sensitivity'),
        ]
        
        all_imported = True
        for module_name, obj_name in critical_modules:
            try:
                module = __import__(module_name, fromlist=[obj_name])
                obj = getattr(module, obj_name)
                self.log_pass(f"  - {module_name}.{obj_name}")
            except Exception as e:
                self.log_fail(f"  - {module_name}.{obj_name}", str(e))
                all_imported = False
        
        return all_imported

    # (remaining checks and methods are identical to the original implementation)


def main():
    """Main entry point."""
    checker = ReadinessChecker()
    
    logging.info("="*80)
    logging.info("GDSEARCH READINESS VERIFICATION")
    logging.info("="*80)
    logging.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Working Directory: {Path.cwd()}")
    logging.info("="*80)
    
    # Run all checks
    checker.check_requirements_file()
    checker.check_gpu_availability()
    checker.check_results_directory()
    checker.check_src_imports()
    checker.check_unit_tests()
    checker.check_integrity_warnings()
    checker.check_golden_test()
    checker.run_dry_run_experiments()
    
    # Generate final report
    ready = checker.generate_report()
    
    sys.exit(0 if ready else 1)


if __name__ == '__main__':
    main()
