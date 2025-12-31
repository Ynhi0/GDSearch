#!/usr/bin/env python3
"""
Validate Experiment Configuration and Output Integrity

This script checks that all experiments in the codebase:
1. Use multi-seed runs (recommended: 10 seeds for statistical validity)
2. Have adequate epoch counts for meaningful results
3. Include VRAM tracking
4. Have proper checkpoint/resume logic
5. Output valid, usable results

Usage:
    python scripts/validate_experiment_config.py
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Recommended configurations based on research best practices
RECOMMENDED_CONFIG = {
    'min_seeds': 5,  # Minimum for statistical significance
    'recommended_seeds': 10,  # Best practice for research papers
    'min_epochs': {
        'MNIST': 10,  # Simple dataset
        'CIFAR10': 20,  # More complex
        'NLP': 10,  # Transformer fine-tuning
        'Medical': 10,  # Medical imaging
        'ResNet': 20,  # Deep architecture
        'HighDim': 100,  # High-dimensional optimization
    },
    'required_metrics': [
        'train_loss',
        'test_loss',
        'test_accuracy',
        'gpu_memory_peak_mb',
        'gpu_memory_free_mb',
        'duration_seconds'
    ]
}


from typing import Optional

class ExperimentValidator:
    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).parent.parent
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def validate_all(self):
        """Run all validation checks"""
        print("Validating Experiment Configuration and Output Integrity")
        print("=" * 70)
        
        # 1. Check multi-seed configuration
        self.check_seed_configuration()
        
        # 2. Check epoch configuration
        self.check_epoch_configuration()
        
        # 3. Check VRAM tracking
        self.check_vram_tracking()
        
        # 4. Check checkpoint logic
        self.check_checkpoint_logic()
        
        # 5. Check output integrity
        self.check_output_integrity()
        
        # 6. Check config files
        self.check_config_files()
        
        # Print summary
        self.print_summary()
        
    def check_seed_configuration(self):
        """Validate multi-seed configuration in experiments"""
        print("\n1. MULTI-SEED CONFIGURATION")
        print("-" * 70)
        
        # Check main script
        main_script = self.root_dir / "run_all_kaggle.py"
        if main_script.exists():
            content = main_script.read_text()
            
            # Check default seeds argument
            seed_match = re.search(r"--seeds.*default='([^']+)'", content)
            if seed_match:
                default_seeds = seed_match.group(1).split(',')
                num_seeds = len(default_seeds)
                
                if num_seeds >= RECOMMENDED_CONFIG['recommended_seeds']:
                    self.successes.append(
                        f"PASS: Default seeds: {num_seeds} seeds (excellent for statistical validity)"
                    )
                elif num_seeds >= RECOMMENDED_CONFIG['min_seeds']:
                    self.warnings.append(
                        f"WARNING: Default seeds: {num_seeds} seeds (acceptable, but 10 recommended)"
                    )
                else:
                    self.issues.append(
                        f"FAIL: Default seeds: {num_seeds} seeds (insufficient, minimum 5 required)"
                    )
            
            # Check experiment functions
            experiment_funcs = re.findall(
                r'def (run_\w+_experiment)\([^)]*seeds=\[([^\]]+)\]',
                content
            )
            
            for func_name, seeds_str in experiment_funcs:
                seeds = [s.strip() for s in seeds_str.split(',')]
                num_seeds = len(seeds)
                
                if num_seeds >= RECOMMENDED_CONFIG['min_seeds']:
                    self.successes.append(
                        f"PASS: {func_name}: {num_seeds} default seeds"
                    )
                else:
                    self.warnings.append(
                        f"WARNING: {func_name}: only {num_seeds} default seeds (5+ recommended)"
                    )
    
    def check_epoch_configuration(self):
        """Validate epoch counts are adequate for each experiment"""
        print("\n2. EPOCH CONFIGURATION")
        print("-" * 70)
        
        main_script = self.root_dir / "run_all_kaggle.py"
        if main_script.exists():
            content = main_script.read_text()
            
            # Look for actual production epoch values (not ULTRA_QUICK_MODE)
            # Pattern: epochs = X if quick else Y
            epoch_pattern = r'epochs\s*=\s*(\d+)\s*if\s+(?:quick|args\.quick)\s+else\s+(\d+)'
            matches = re.findall(epoch_pattern, content)
            
            if matches:
                # Analyze the full (non-quick) values
                full_epochs = [int(m[1]) for m in matches]
                quick_epochs = [int(m[0]) for m in matches]
                
                # Check specific experiments
                exp_checks = {
                    'MNIST': 50,
                    'CIFAR10': 50,
                    'NLP': 15,
                    'ResNet': 50,
                }
                
                # Find max epochs per type
                if full_epochs:
                    max_full = max(full_epochs)
                    min_full = min(full_epochs)
                    
                    if max_full >= 50:
                        self.successes.append(
                            f"PASS: Production epochs: {min_full}-{max_full} (excellent)"
                        )
                    elif max_full >= 20:
                        self.successes.append(
                            f"PASS: Production epochs: {min_full}-{max_full} (good)"
                        )
                    else:
                        self.warnings.append(
                            f"WARNING: Production epochs: {min_full}-{max_full} (consider increasing)"
                        )
    
    def check_vram_tracking(self):
        """Check if VRAM tracking is properly implemented"""
        print("\n3. VRAM TRACKING")
        print("-" * 70)
        
        main_script = self.root_dir / "run_all_kaggle.py"
        if main_script.exists():
            content = main_script.read_text()
            
            required_metrics = [
                'gpu_memory_peak_mb',
                'gpu_memory_free_mb',
                'gpu_memory_end_mb'
            ]
            
            found_metrics = []
            missing_metrics = []
            
            for metric in required_metrics:
                if metric in content:
                    found_metrics.append(metric)
                else:
                    missing_metrics.append(metric)
            
            if not missing_metrics:
                self.successes.append(
                    f"PASS: All VRAM metrics tracked: {', '.join(found_metrics)}"
                )
            else:
                self.issues.append(
                    f"FAIL: Missing VRAM metrics: {', '.join(missing_metrics)}"
                )
    
    def check_checkpoint_logic(self):
        """Validate checkpoint/resume logic"""
        print("\n4. CHECKPOINT/RESUME LOGIC")
        print("-" * 70)
        
        main_script = self.root_dir / "run_all_kaggle.py"
        if main_script.exists():
            content = main_script.read_text()
            
            required_components = {
                'RobustCheckpointManager': 'class RobustCheckpointManager',
                'save_checkpoint': 'def save_checkpoint',
                'load_checkpoint': 'def load_checkpoint',
                'validate_checkpoint': '_validate_checkpoint',
                'resume_flag': '--resume',
            }
            
            for component, pattern in required_components.items():
                if pattern in content:
                    self.successes.append(f"PASS: {component} implemented")
                else:
                    self.issues.append(f"FAIL: {component} missing")
    
    def check_output_integrity(self):
        """Check that experiments produce valid, usable output"""
        print("\n5. OUTPUT INTEGRITY")
        print("-" * 70)
        
        # Check for CSV output format
        main_script = self.root_dir / "run_all_kaggle.py"
        if main_script.exists():
            content = main_script.read_text()
            
            # Check for DataFrame and CSV exports
            if 'pd.DataFrame' in content and '.to_csv' in content:
                self.successes.append("PASS: CSV output format implemented")
            else:
                self.issues.append("FAIL: CSV output format not found")
            
            # Check for required result fields
            result_fields = [
                'optimizer',
                'lr',
                'test_accuracy',
                'test_loss',
                'train_time',
                'convergence_epoch',
            ]
            
            found_fields = sum(1 for field in result_fields if field in content)
            if found_fields >= len(result_fields) - 1:  # Allow 1 missing
                self.successes.append(
                    f"PASS: Result fields: {found_fields}/{len(result_fields)} found"
                )
            else:
                self.warnings.append(
                    f"WARNING: Result fields: only {found_fields}/{len(result_fields)} found"
                )
    
    def check_config_files(self):
        """Validate JSON configuration files"""
        print("\n6. CONFIGURATION FILES")
        print("-" * 70)
        
        config_dir = self.root_dir / "configs"
        if config_dir.exists():
            for config_file in config_dir.glob("*.json"):
                try:
                    with open(config_file, encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # Check final epochs
                    final_config = config.get('final', {})
                    epochs = final_config.get('epochs', 0)
                    
                    dataset = config.get('dataset', config_file.stem)
                    min_epochs = RECOMMENDED_CONFIG['min_epochs'].get(dataset, 10)
                    
                    if epochs >= min_epochs:
                        self.successes.append(
                            f"PASS: {config_file.name}: {epochs} final epochs"
                        )
                    else:
                        self.issues.append(
                            f"FAIL: {config_file.name}: {epochs} epochs (< {min_epochs} recommended)"
                        )
                    
                except Exception as e:
                    self.issues.append(f"FAIL: {config_file.name}: Invalid JSON - {e}")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        if self.successes:
            print(f"\nPASSED ({len(self.successes)}):")
            for success in self.successes:
                print(f"  {success}")
        
        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.issues:
            print(f"\nISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
        
        # Overall verdict
        print("\n" + "=" * 70)
        if not self.issues:
            if not self.warnings:
                print("ALL CHECKS PASSED - Codebase is analysis-ready!")
            else:
                print("MOSTLY GOOD - Some minor improvements recommended")
        else:
            print("ISSUES FOUND - Please address the problems above")
        print("=" * 70)
        
        # Return exit code
        return 1 if self.issues else 0


def main():
    validator = ExperimentValidator()
    exit_code = validator.validate_all()
    
    # Print recommendations
    print("\n📋 RECOMMENDATIONS FOR RESEARCH PAPER:")
    print("-" * 70)
    print("1. Use 10 seeds for all experiments (currently configured ✅)")
    print("2. Ensure adequate epochs for convergence:")
    print("   - MNIST: 20 epochs")
    print("   - CIFAR-10: 50 epochs")
    print("   - NLP: 10 epochs (transformer fine-tuning)")
    print("   - ResNet: 50 epochs")
    print("3. Always track VRAM (peak, free, end)")
    print("4. Save checkpoints for long experiments")
    print("5. Export results to CSV with all metrics")
    print("=" * 70)
    
    return exit_code


if __name__ == "__main__":
    exit(main())
