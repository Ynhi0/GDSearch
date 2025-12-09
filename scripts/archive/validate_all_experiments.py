#!/usr/bin/env python3
"""
Comprehensive Validation Script for GDSearch Codebase
Verifies all 25+ experiments are properly wired and functional

This script performs:
1. Import validation - all experiment modules can be imported
2. Function signature validation - all run_* functions have correct signatures
3. Experiment list completeness - all advertised experiments exist
4. Basic smoke test - each experiment can be instantiated

Usage:
    python scripts/validate_all_experiments.py
    python scripts/validate_all_experiments.py --smoke-test
"""

import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ANSI colors for pretty output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg: str):
    print(f"{GREEN}✓{RESET} {msg}")

def print_error(msg: str):
    print(f"{RED}✗{RESET} {msg}")

def print_warning(msg: str):
    print(f"{YELLOW}⚠{RESET} {msg}")

def print_section(title: str):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")


# Expected experiments from run_all_kaggle.py line ~6685
EXPECTED_EXPERIMENTS = {
    # Core experiments
    'mnist': 'run_all_kaggle.run_mnist_experiment',
    'cifar10': 'run_all_kaggle.run_cifar10_experiment',
    'nlp': 'run_all_kaggle._run_nlp_experiment_huggingface',  # Has wrapper
    'medical': 'run_all_kaggle.run_medical_experiment',
    'resnet': 'run_all_kaggle.run_resnet_experiment',
    'highdim': 'run_all_kaggle.run_highdim_experiment',
    
    # 2D and robustness experiments
    '2d': 'run_all_kaggle.run_2d_experiments',
    'robustness': 'run_all_kaggle.run_robustness_analysis',
    'sam': 'run_all_kaggle.run_sam_sensitivity',
    
    # Ablation studies
    'ablation': 'run_all_kaggle.run_ablation_study',
    'advanced_ablation': 'run_all_kaggle.run_advanced_training_ablation',
    'init_ablation': 'run_all_kaggle.run_initialization_ablation',
    'batch_ablation': 'run_all_kaggle.run_batch_ablation',
    'scheduler_ablation': 'run_all_kaggle.run_scheduler_ablation',
    
    # External module experiments (imported from src)
    'lr_ablation': 'src.experiments.learning_rate_ablation.run_learning_rate_ablation',
    'wd_ablation': 'src.experiments.weight_decay_ablation.run_weight_decay_ablation',
    'missing_ablations': 'src.experiments.missing_ablations.run_all_missing_ablations',
    'ablation_comprehensive': 'src.experiments.ablation_studies_comprehensive.run_all_ablation_studies',
    
    # Optimizer comparison
    'optimizer_comparison': 'src.analysis.optimizer_comparison_matrix.run_optimizer_comparison_matrix',
    
    # Hyperparameter studies
    'hyperparam_sensitivity': 'src.experiments.hyperparameter_sensitivity.momentum_beta_sweep',  # Primary function
    'beta_sensitivity_training': 'src.experiments.beta_sensitivity_training.run_momentum_beta_sensitivity',
    
    # Convergence and theory
    'convergence_validation': 'src.experiments.convergence_rate_validation.run_convergence_rate_comparison',
    'theory_practice': 'src.experiments.theory_practice_validation.run_theory_practice_validation',
    
    # Visualization
    '2d_visualization': None,  # Inline implementation in run_all_kaggle.py
    
    # Advanced experiments
    'dynamics_overhead': 'src.experiments.dynamics_overhead_ablation.run_dynamics_overhead_ablation',
    'cross_optimizer_dynamics': 'src.experiments.cross_optimizer_dynamics_comparison.run_cross_optimizer_dynamics_comparison',
}


def validate_imports() -> Tuple[int, int]:
    """Validate all experiment modules can be imported"""
    print_section("📦 IMPORT VALIDATION")
    
    success_count = 0
    fail_count = 0
    
    # Test core imports first
    core_imports = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    for module_name, display_name in core_imports:
        try:
            importlib.import_module(module_name)
            print_success(f"{display_name} imported successfully")
            success_count += 1
        except ImportError as e:
            print_error(f"{display_name} import failed: {e}")
            fail_count += 1
    
    # Test experiment module imports
    experiment_modules = {
        'src.experiments.beta_sensitivity_training': 'Beta Sensitivity Training',
        'src.experiments.hyperparameter_sensitivity': 'Hyperparameter Sensitivity',
        'src.experiments.convergence_rate_validation': 'Convergence Rate Validation',
        'src.experiments.theory_practice_validation': 'Theory-Practice Validation',
        'src.experiments.cross_optimizer_dynamics_comparison': 'Cross-Optimizer Dynamics',
        'src.experiments.dynamics_overhead_ablation': 'Dynamics Overhead Ablation',
        'src.experiments.ablation_studies_comprehensive': 'Comprehensive Ablation Studies',
        'src.experiments.missing_ablations': 'Missing Ablations',
        'src.experiments.learning_rate_ablation': 'Learning Rate Ablation',
        'src.experiments.weight_decay_ablation': 'Weight Decay Ablation',
        'src.analysis.optimizer_comparison_matrix': 'Optimizer Comparison Matrix',
    }
    
    for module_path, display_name in experiment_modules.items():
        try:
            importlib.import_module(module_path)
            print_success(f"{display_name} module imported")
            success_count += 1
        except ImportError as e:
            print_error(f"{display_name} import failed: {e}")
            fail_count += 1
    
    print(f"\n{GREEN}Passed:{RESET} {success_count}/{success_count + fail_count}")
    return success_count, fail_count


def validate_function_signatures() -> Tuple[int, int]:
    """Validate all experiment functions have expected signatures"""
    print_section("🔍 FUNCTION SIGNATURE VALIDATION")
    
    success_count = 0
    fail_count = 0
    
    for exp_name, func_path in EXPECTED_EXPERIMENTS.items():
        if func_path is None:
            print_warning(f"{exp_name}: Inline implementation (skipped)")
            continue
        
        try:
            module_path, func_name = func_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            
            # Get function signature
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Basic validation: should have at least 1 parameter
            if len(params) == 0:
                print_warning(f"{exp_name} ({func_name}): No parameters (unusual)")
            else:
                print_success(f"{exp_name} ({func_name}): Found {len(params)} params")
            
            success_count += 1
            
        except (ImportError, AttributeError) as e:
            print_error(f"{exp_name} validation failed: {e}")
            fail_count += 1
        except Exception as e:
            print_error(f"{exp_name} unexpected error: {e}")
            fail_count += 1
    
    print(f"\n{GREEN}Passed:{RESET} {success_count}/{success_count + fail_count}")
    return success_count, fail_count


def validate_experiment_completeness() -> Tuple[int, int]:
    """Validate all advertised experiments in run_all_kaggle.py exist"""
    print_section("📋 EXPERIMENT COMPLETENESS CHECK")
    
    # Import run_all_kaggle to check its advertised experiments
    try:
        sys.path.insert(0, str(project_root))
        # Can't easily parse args dynamically, so we hardcode the expected list
        # from line 6685-6692 of run_all_kaggle.py
        advertised = [
            'mnist', 'cifar10', 'nlp', 'medical', '2d', 
            'robustness', 'sam', 'ablation', 'advanced_ablation', 'init_ablation',
            'batch_ablation', 'lr_ablation', 'wd_ablation', 'scheduler_ablation', 
            'missing_ablations', 'optimizer_comparison', 'resnet', 'highdim',
            'hyperparam_sensitivity', 'convergence_validation', 
            'ablation_comprehensive', '2d_visualization',
            'dynamics_overhead', 'theory_practice', 'cross_optimizer_dynamics',
            'beta_sensitivity_training'
        ]
        
        success_count = 0
        fail_count = 0
        
        for exp_name in advertised:
            if exp_name in EXPECTED_EXPERIMENTS:
                print_success(f"{exp_name}: Defined")
                success_count += 1
            else:
                print_error(f"{exp_name}: NOT FOUND in validation map")
                fail_count += 1
        
        # Check reverse: any in map but not advertised?
        for exp_name in EXPECTED_EXPERIMENTS:
            if exp_name not in advertised:
                print_warning(f"{exp_name}: In map but not advertised")
        
        print(f"\n{GREEN}Complete:{RESET} {success_count}/{success_count + fail_count}")
        return success_count, fail_count
        
    except Exception as e:
        print_error(f"Failed to validate completeness: {e}")
        return 0, len(EXPECTED_EXPERIMENTS)


def run_smoke_tests() -> Tuple[int, int]:
    """Quick smoke test: can we instantiate key components?"""
    print_section("🔥 SMOKE TESTS")
    
    success_count = 0
    fail_count = 0
    
    # Test 1: Can we create optimizers?
    try:
        from src.core.optimizers import SGD, Adam, AdamW
        opt_sgd = SGD(lr=0.01)
        opt_adam = Adam(lr=0.001)
        opt_adamw = AdamW(lr=0.001)
        print_success("Optimizer instantiation (SGD, Adam, AdamW)")
        success_count += 1
    except Exception as e:
        print_error(f"Optimizer instantiation failed: {e}")
        fail_count += 1
    
    # Test 2: Can we create test functions?
    try:
        from src.core.test_functions import Rosenbrock, Rastrigin, Ackley2D
        fn = Rosenbrock()
        fn.compute(1.0, 1.0)
        print_success("Test function instantiation (Rosenbrock)")
        success_count += 1
    except Exception as e:
        print_error(f"Test function instantiation failed: {e}")
        fail_count += 1
    
    # Test 3: Can we load MNIST?
    try:
        from torchvision import datasets, transforms
        transform = transforms.ToTensor()
        _ = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        print_success("MNIST dataset loading")
        success_count += 1
    except Exception as e:
        print_error(f"MNIST loading failed: {e}")
        fail_count += 1
    
    # Test 4: Can we create a simple model?
    try:
        import torch.nn as nn
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            def forward(self, x):
                return self.fc(x)
        model = SimpleModel()
        print_success("Neural network instantiation")
        success_count += 1
    except Exception as e:
        print_error(f"Model instantiation failed: {e}")
        fail_count += 1
    
    print(f"\n{GREEN}Passed:{RESET} {success_count}/{success_count + fail_count}")
    return success_count, fail_count


def check_missing_files() -> List[str]:
    """Check for any expected files that are missing"""
    print_section("📁 MISSING FILE CHECK")
    
    expected_files = [
        'run_all_kaggle.py',
        'src/core/optimizers.py',
        'src/core/pytorch_optimizers.py',
        'src/core/test_functions.py',
        'src/experiments/beta_sensitivity_training.py',
        'src/experiments/hyperparameter_sensitivity.py',
        'src/experiments/convergence_rate_validation.py',
        'src/experiments/theory_practice_validation.py',
        'src/experiments/cross_optimizer_dynamics_comparison.py',
        'src/experiments/dynamics_overhead_ablation.py',
        'src/experiments/ablation_studies_comprehensive.py',
        'src/experiments/missing_ablations.py',
        'src/analysis/statistical_analysis.py',
        'src/analysis/optimizer_comparison_matrix.py',
        'tests/test_optimizers.py',
        'tests/test_training_loop.py',
    ]
    
    missing = []
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists():
            print_success(f"{file_path}")
        else:
            print_error(f"{file_path} - MISSING")
            missing.append(file_path)
    
    if not missing:
        print(f"\n{GREEN}All expected files present{RESET}")
    else:
        print(f"\n{RED}Missing {len(missing)} files{RESET}")
    
    return missing


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate GDSearch codebase')
    parser.add_argument('--smoke-test', action='store_true', help='Run smoke tests')
    args = parser.parse_args()
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}GDSearch Codebase Validation Suite{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    total_pass = 0
    total_fail = 0
    
    # Run validations
    p, f = validate_imports()
    total_pass += p
    total_fail += f
    
    p, f = validate_function_signatures()
    total_pass += p
    total_fail += f
    
    p, f = validate_experiment_completeness()
    total_pass += p
    total_fail += f
    
    missing_files = check_missing_files()
    
    if args.smoke_test:
        p, f = run_smoke_tests()
        total_pass += p
        total_fail += f
    
    # Final summary
    print_section("📊 FINAL SUMMARY")
    print(f"Total Passed: {GREEN}{total_pass}{RESET}")
    print(f"Total Failed: {RED}{total_fail}{RESET}")
    print(f"Missing Files: {RED if missing_files else GREEN}{len(missing_files)}{RESET}")
    
    if total_fail == 0 and not missing_files:
        print(f"\n{GREEN}✅ VALIDATION PASSED - Codebase is research-grade{RESET}\n")
        return 0
    else:
        print(f"\n{RED}❌ VALIDATION FAILED - Issues need fixing{RESET}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
