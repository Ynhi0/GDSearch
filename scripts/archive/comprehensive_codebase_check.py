#!/usr/bin/env python3
"""
Comprehensive Codebase Health Check

Scans the entire codebase for:
1. Missing save/checkpoint/resume logic
2. Output integrity issues
3. Error-prone patterns
4. Missing VRAM tracking
5. Inadequate epoch/seed configurations

Usage:
    python scripts/comprehensive_codebase_check.py
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class CodebaseHealthChecker:
    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        self.info = []

    def check_all(self):
        """Run all health checks"""
        logger.info("COMPREHENSIVE CODEBASE HEALTH CHECK")
        logger.info("=" * 80)

        # Check main experiment files
        self.check_checkpoint_resume_logic()
        self.check_vram_tracking()
        self.check_output_integrity()
        self.check_error_handling()
        self.check_seed_epoch_config()

        # Print summary
        self.print_summary()

    def check_checkpoint_resume_logic(self):
        """Check for proper checkpoint/resume implementation"""
        logger.info("\n1. CHECKPOINT/RESUME LOGIC")
        logger.info("-" * 80)

        main_file = self.root_dir / "run_all_kaggle.py"
        if not main_file.exists():
            self.errors.append("❌ Main file run_all_kaggle.py not found")
            return

        content = main_file.read_text()

        # Check for checkpoint manager
        if "class RobustCheckpointManager" in content:
            self.info.append("✅ RobustCheckpointManager class found")
        else:
            self.errors.append("❌ RobustCheckpointManager class missing")

        # Check for save/load methods
        patterns = {
            'save_checkpoint': r'def save_checkpoint\(',
            'load_checkpoint': r'def load_checkpoint\(',
            'validate_checkpoint': r'def _validate_checkpoint\(',
            'resume_flag': r'--resume',
            'checkpoint_usage': r'checkpoint_manager\.save_checkpoint\(',
        }

        for name, pattern in patterns.items():
            if re.search(pattern, content):
                self.info.append(f"✅ {name} implemented")
            else:
                self.warnings.append(f"⚠️  {name} may not be fully implemented")

        # Check experiment functions use checkpoints
        experiment_funcs = re.findall(r'def (run_\w+_experiment)\(', content)
        for func in experiment_funcs:
            if f"{func}.*checkpoint_manager" in content or "checkpoint" in content[content.find(func):content.find(func)+5000]:
                self.info.append(f"✅ {func} supports checkpointing")
            else:
                self.warnings.append(f"⚠️  {func} may not use checkpointing")

    def check_vram_tracking(self):
        """Check for VRAM tracking in all experiments"""
        logger.info("\n2. VRAM TRACKING")
        logger.info("-" * 80)

        main_file = self.root_dir / "run_all_kaggle.py"
        content = main_file.read_text()

        vram_metrics = [
            'gpu_memory_peak_mb',
            'gpu_memory_free_mb',
            'gpu_memory_end_mb',
            'torch.cuda.max_memory_allocated',
            'torch.cuda.memory_allocated',
        ]

        found_metrics = []
        for metric in vram_metrics:
            if metric in content:
                found_metrics.append(metric)

        if len(found_metrics) >= 3:
            self.info.append(f"✅ VRAM tracking comprehensive: {len(found_metrics)} metrics")
        elif found_metrics:
            self.warnings.append(f"⚠️  Partial VRAM tracking: {len(found_metrics)} metrics")
        else:
            self.errors.append("❌ No VRAM tracking found")

        # Check if VRAM is saved to CSV
        if 'gpu_memory' in content and 'to_csv' in content:
            self.info.append("✅ VRAM metrics saved to CSV")
        else:
            self.warnings.append("⚠️  VRAM metrics may not be saved to CSV")

    def check_output_integrity(self):
        """Check that outputs are properly structured and saved"""
        logger.info("\n3. OUTPUT INTEGRITY")
        logger.info("-" * 80)

        main_file = self.root_dir / "run_all_kaggle.py"
        content = main_file.read_text()

        # Check for DataFrame usage
        if 'pd.DataFrame' in content:
            self.info.append("✅ Using pandas DataFrame for results")
        else:
            self.warnings.append("⚠️  Not using pandas DataFrame")

        # Check for CSV export
        csv_exports = len(re.findall(r'\.to_csv\(', content))
        if csv_exports > 0:
            self.info.append(f"✅ {csv_exports} CSV export points found")
        else:
            self.errors.append("❌ No CSV exports found")

        # Check for required result fields
        required_fields = [
            'optimizer',
            'lr',
            'test_accuracy',
            'test_loss',
            'train_time',
            'epoch',
        ]

        found_fields = sum(1 for field in required_fields if field in content)
        if found_fields >= len(required_fields) - 1:
            self.info.append(f"✅ Result fields: {found_fields}/{len(required_fields)}")
        else:
            self.warnings.append(f"⚠️  Missing result fields: {found_fields}/{len(required_fields)}")

        # Check for MLflow logging (optional but good)
        if 'mlflow.log_metric' in content or 'mlflow.log_param' in content:
            self.info.append("✅ MLflow logging implemented")
        else:
            self.info.append("ℹ️  MLflow logging not used (optional)")

    def check_error_handling(self):
        """Check for proper error handling"""
        logger.info("\n4. ERROR HANDLING")
        logger.info("-" * 80)

        python_files = list(self.root_dir.glob("**/*.py"))

        total_try_blocks = 0
        files_with_try = 0
        bare_except_count = 0

        for py_file in python_files:
            if 'test_' in py_file.name or '__pycache__' in str(py_file):
                continue

            try:
                content = py_file.read_text()

                # Count try blocks
                try_count = content.count('try:')
                if try_count > 0:
                    total_try_blocks += try_count
                    files_with_try += 1

                # Check for bare except
                if re.search(r'except\s*:', content):
                    bare_except_count += 1
                    self.warnings.append(f"⚠️  Bare except in {py_file.name}")
            except (IOError, UnicodeDecodeError) as e:
                # Handle file reading errors
                logging.debug(f"Failed to read {py_file.name}: {e}")
                pass

        self.info.append(f"✅ Error handling: {total_try_blocks} try blocks in {files_with_try} files")
        if bare_except_count > 0:
            self.warnings.append(f"⚠️  {bare_except_count} files with bare except (should specify exception type)")

    def check_seed_epoch_config(self):
        """Check seed and epoch configurations"""
        logger.info("\n5. SEED & EPOCH CONFIGURATION")
        logger.info("-" * 80)

        main_file = self.root_dir / "run_all_kaggle.py"
        content = main_file.read_text()

        # Check default seeds
        seed_match = re.search(r"--seeds.*default='([^']+)'", content)
        if seed_match:
            seeds = seed_match.group(1).split(',')
            if len(seeds) >= 10:
                self.info.append(f"✅ Default seeds: {len(seeds)} (excellent)")
            elif len(seeds) >= 5:
                self.warnings.append(f"⚠️  Default seeds: {len(seeds)} (acceptable, 10 recommended)")
            else:
                self.errors.append(f"❌ Default seeds: {len(seeds)} (insufficient)")

        # Check epoch configurations
        epoch_patterns = {
            'MNIST': r'mnist.*?epochs\s*=\s*(\d+)',
            'CIFAR': r'cifar.*?epochs\s*=\s*(\d+)',
            'NLP': r'nlp.*?epochs\s*=\s*(\d+)',
        }

        for exp_name, pattern in epoch_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                epochs_list = [int(e) for e in matches]
                max_epochs = max(epochs_list)
                if max_epochs >= 20:
                    self.info.append(f"✅ {exp_name}: up to {max_epochs} epochs")
                elif max_epochs >= 10:
                    self.warnings.append(f"⚠️  {exp_name}: up to {max_epochs} epochs (20+ recommended)")
                else:
                    self.warnings.append(f"⚠️  {exp_name}: only {max_epochs} epochs")

    def print_summary(self):
        """Print comprehensive summary"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 HEALTH CHECK SUMMARY")
        logger.info("=" * 80)

        if self.info:
            logger.info(f"\n✅ GOOD ({len(self.info)}):")
            for item in self.info:
                logger.info(f"  {item}")

        if self.warnings:
            logger.info(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for item in self.warnings:
                logger.info(f"  {item}")

        if self.errors:
            logger.info(f"\n❌ ERRORS ({len(self.errors)}):")
            for item in self.errors:
                logger.info(f"  {item}")

        logger.info("\n" + "=" * 80)
        if not self.errors:
            if not self.warnings:
                logger.info("✅ EXCELLENT - Codebase is in great shape!")
            else:
                logger.info("⚠️  GOOD - Minor improvements recommended")
        else:
            logger.info("❌ NEEDS ATTENTION - Please fix errors above")
        logger.info("=" * 80)


def main():
    checker = CodebaseHealthChecker()
    checker.check_all()

    # Return exit code
    return 1 if checker.errors else 0


if __name__ == "__main__":
    exit(main())
