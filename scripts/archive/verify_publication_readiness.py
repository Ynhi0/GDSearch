#!/usr/bin/env python3
"""
Publication Readiness Verification Script

This script performs comprehensive end-to-end checks to verify the codebase
is ready for scientific publication and Kaggle deployment.

Addresses Phase 7 recommendation from the audit report.
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


class PublicationReadinessChecker:
    """Comprehensive publication readiness checker."""
    
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
    
    def check_unit_tests(self):
        """Run unit tests."""
        logging.info("\n" + "="*80)
        logging.info("CHECK 5: Unit Tests")
        logging.info("="*80)
        
        if not Path('tests').exists():
            self.log_warning("Unit tests", "tests/ directory not found")
            return False
        
        try:
            result = subprocess.run(
                ['pytest', 'tests/', '-v', '--tb=short', '-x'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Count passed tests
                passed = result.stdout.count(' PASSED')
                self.log_pass(f"Unit tests passed ({passed} tests)")
                return True
            else:
                self.log_fail("Unit tests", "Some tests failed")
                logging.error(result.stdout[-500:])  # Last 500 chars
                return False
        except FileNotFoundError:
            self.log_warning("pytest", "Not installed - skipping unit tests")
            return False
        except subprocess.TimeoutExpired:
            self.log_fail("Unit tests", "Timeout (>5min)")
            return False
    
    def run_dry_run_experiments(self):
        """Run 1-epoch dry run on ALL experiments."""
        logging.info("\n" + "="*80)
        logging.info("CHECK 6: Dry Run (1 epoch ALL experiments)")
        logging.info("="*80)
        
        experiments = [
            'mnist',
            'cifar10',
            'nlp',
            '2d',
            'highdim',
            'beta_sensitivity_training'
        ]
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        all_passed = True
        for exp_name in experiments:
            logging.info(f"\nDry run: {exp_name.upper()}")
            
            try:
                # Run 1 epoch with minimal config
                cmd = [
                    sys.executable,
                    'run_all_kaggle.py',
                    '--experiments', exp_name,
                    '--ultra-quick',  # 2 epochs, minimal optimizers
                    '--seeds', '1',
                    '--skip-tuning',
                    '--results-dir', 'results/.dry_run_test'
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes max per experiment
                )
                
                if result.returncode == 0:
                    self.log_pass(f"  - {exp_name}: dry run successful")
                else:
                    self.log_fail(f"  - {exp_name}", "Dry run failed")
                    logging.error(f"    STDERR: {result.stderr[-200:]}")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                self.log_fail(f"  - {exp_name}", "Timeout (>10min)")
                all_passed = False
            except Exception as e:
                self.log_fail(f"  - {exp_name}", str(e))
                all_passed = False
        
        # Cleanup dry run results
        import shutil
        dry_run_dir = Path('results/.dry_run_test')
        if dry_run_dir.exists():
            shutil.rmtree(dry_run_dir)
            logging.info("\nCleaned up dry run artifacts")
        
        return all_passed
    
    def check_golden_test(self):
        """Run --verify-resume golden test."""
        logging.info("\n" + "="*80)
        logging.info("CHECK 7: Golden Test (Resume Determinism)")
        logging.info("="*80)
        
        try:
            result = subprocess.run(
                [sys.executable, 'run_all_kaggle.py', '--verify-resume'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if 'GOLDEN TEST PASSED' in result.stdout:
                self.log_pass("Golden test: Train(10) == Train(5)→Save→Load→Train(5)")
                return True
            else:
                self.log_fail("Golden test", "Resume produces different weights")
                return False
        except Exception as e:
            self.log_fail("Golden test", str(e))
            return False
    
    def check_scientific_integrity_warnings(self):
        """Check OOM recovery warnings are present."""
        logging.info("\n" + "="*80)
        logging.info("CHECK 8: Scientific Integrity Warnings")
        logging.info("="*80)
        
        run_all_path = Path('run_all_kaggle.py')
        with open(run_all_path, encoding='utf-8') as f:
            content = f.read()
        
        # Check for OOM warnings
        warning_count = content.count('SCIENTIFIC INTEGRITY: This run is INVALID')
        
        if warning_count >= 3:
            self.log_pass(f"OOM recovery warnings present ({warning_count} locations)")
        else:
            self.log_fail("OOM warnings", f"Found {warning_count}, expected ≥3")
            return False
        
        # Check SelfHealingTrainer docstring
        enhancements_path = Path('src/core/training_enhancements.py')
        if enhancements_path.exists():
            with open(enhancements_path, encoding='utf-8') as f:
                enhancements_content = f.read()
            
            if 'SCIENTIFIC INTEGRITY WARNING' in enhancements_content:
                self.log_pass("SelfHealingTrainer docstring warning present")
            else:
                self.log_warning("SelfHealingTrainer", "Missing docstring warning")
        
        return True
    
    def generate_report(self):
        """Generate final report."""
        logging.info("\n" + "="*80)
        logging.info("PUBLICATION READINESS REPORT")
        logging.info("="*80)
        
        total_checks = len(self.checks_passed) + len(self.checks_failed)
        pass_rate = len(self.checks_passed) / total_checks * 100 if total_checks > 0 else 0
        
        logging.info(f"\nChecks Passed: {len(self.checks_passed)}")
        logging.info(f"Checks Failed: {len(self.checks_failed)}")
        logging.info(f"Warnings: {len(self.warnings)}")
        logging.info(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.checks_failed:
            logging.info("\nFAILED CHECKS:")
            for check, reason in self.checks_failed:
                logging.info(f"   - {check}: {reason}")
        
        if self.warnings:
            logging.info("\nWARNINGS:")
            for check, reason in self.warnings:
                logging.info(f"   - {check}: {reason}")
        
        # Save report
        report_path = Path('results/publication_readiness_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'passed': self.checks_passed,
            'failed': [{'check': c, 'reason': r} for c, r in self.checks_failed],
            'warnings': [{'check': c, 'reason': r} for c, r in self.warnings],
            'pass_rate': pass_rate,
            'verdict': 'READY' if len(self.checks_failed) == 0 else 'NOT READY'
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"\nReport saved to {report_path}")
        
        # Final verdict
        logging.info("\n" + "="*80)
        if len(self.checks_failed) == 0:
            logging.info("VERDICT: PUBLICATION-READY")
            logging.info("="*80)
            logging.info("Codebase is ready for:")
            logging.info("  Academic thesis defense")
            logging.info("  Peer-reviewed journal publication")
            logging.info("  Reproducible research benchmarks")
            logging.info("  Kaggle GPU deployment")
        else:
            logging.info("VERDICT: NOT READY")
            logging.info("="*80)
            logging.info(f"Fix {len(self.checks_failed)} critical issues before publication")
        logging.info("="*80)
        
        return len(self.checks_failed) == 0


def main():
    """Main entry point."""
    checker = PublicationReadinessChecker()
    
    logging.info("="*80)
    logging.info("GDSEARCH PUBLICATION READINESS VERIFICATION")
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
    checker.check_scientific_integrity_warnings()
    checker.check_golden_test()
    checker.run_dry_run_experiments()
    
    # Generate final report
    ready = checker.generate_report()
    
    sys.exit(0 if ready else 1)


if __name__ == '__main__':
    main()
