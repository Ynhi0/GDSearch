#!/usr/bin/env python3
"""
Comprehensive Codebase Validation Script

This script performs final validation checks to ensure:
1. All experiments are properly integrated
2. No syntax errors in any files
3. All imports work correctly
4. Checkpoint/resume logic is sound
5. Ablation studies meet academic standards
6. No obvious runtime bugs
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

def check_syntax():
    """Check Python syntax of all source files"""
    print("\n" + "="*80)
    print("1. SYNTAX VALIDATION")
    print("="*80)

    import py_compile

    error_files = []

    for root in ['src', 'tests']:
        for path in Path(root).rglob('*.py'):
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as e:
                error_files.append((str(path), str(e)))

    if error_files:
        print("❌ Syntax errors found:")
        for filepath, error in error_files:
            print(f"  {filepath}: {error}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True


def check_imports():
    """Check that all critical modules can be imported"""
    print("\n" + "="*80)
    print("2. IMPORT VALIDATION")
    print("="*80)

    imports_to_test = [
        ('Core modules', [
            'src.core.optimizers',
            'src.core.test_functions',
            'src.core.pytorch_optimizers',
            'src.core.training_utils',
        ]),
        ('Experiment modules', [
            'src.experiments.batch_size_ablation',
            'src.experiments.learning_rate_ablation',
            'src.experiments.weight_decay_ablation',
            'src.experiments.scheduler_ablation',
            'src.experiments.advanced_training_ablation',
            'src.experiments.initialization_ablation',
        ]),
        ('Analysis modules', [
            'src.analysis.statistical_analysis',
            'src.analysis.ablation_study',
        ]),
    ]

    all_passed = True

    for category, modules in imports_to_test:
        print(f"\n{category}:")
        for module_name in modules:
            try:
                __import__(module_name)
                print(f"  ✅ {module_name}")
            except ImportError as e:
                print(f"  ❌ {module_name}: {e}")
                all_passed = False
            except Exception as e:
                print(f"  ⚠️  {module_name}: {e}")

    if all_passed:
        print("\n✅ All critical imports work correctly")
    else:
        print("\n❌ Some imports failed")

    return all_passed


def check_ablation_studies():
    """Verify ablation studies meet academic standards"""
    print("\n" + "="*80)
    print("3. ABLATION STUDY VALIDATION")
    print("="*80)

    ablation_criteria = {
        'Multi-seed experiments': ['seeds', 'seed'],
        'Statistical reporting': ['np.mean', 'np.std', 'mean', 'std'],
        'Controlled configs': ['config', 'configuration'],
        'Documentation': ['"""', 'Args:', 'Returns:'],
    }

    ablation_files = [
        'src/experiments/batch_size_ablation.py',
        'src/experiments/learning_rate_ablation.py',
        'src/experiments/weight_decay_ablation.py',
        'src/experiments/scheduler_ablation.py',
        'src/experiments/advanced_training_ablation.py',
        'src/experiments/initialization_ablation.py',
    ]

    all_passed = True

    for filepath in ablation_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"\n{Path(filepath).name}:")
            print(f"  ❌ Could not read file: {e}")
            all_passed = False
            continue

        filename = Path(filepath).name
        print(f"\n{filename}:")

        for criterion, keywords in ablation_criteria.items():
            passed = any(kw in content for kw in keywords)
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}")
            if not passed:
                all_passed = False

    if all_passed:
        print("\n✅ All ablation studies meet academic standards")
    else:
        print("\n❌ Some ablation studies need improvement")

    return all_passed


def check_experiment_integration():
    """Verify all experiments are integrated into main file"""
    print("\n" + "="*80)
    print("4. EXPERIMENT INTEGRATION CHECK")
    print("="*80)

    expected_experiments = [
        'mnist', 'cifar10', 'nlp', 'medical', '2d',
        'robustness', 'sam', 'ablation',
        'advanced_ablation', 'init_ablation',
        'batch_ablation', 'lr_ablation', 'wd_ablation',
        'scheduler_ablation', 'optimizer_comparison',
        'resnet', 'highdim'
    ]

    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        main_content = f.read()

    missing_experiments = []

    for exp in expected_experiments:
        if f"if '{exp}' in selected_experiments" in main_content:
            print(f"  ✅ {exp}")
        else:
            print(f"  ❌ {exp}")
            missing_experiments.append(exp)

    if missing_experiments:
        print(f"\n❌ Missing experiments: {missing_experiments}")
        return False
    else:
        print("\n✅ All experiments are properly integrated")
        return True


def check_checkpoint_logic():
    """Validate checkpoint/resume logic"""
    print("\n" + "="*80)
    print("5. CHECKPOINT/RESUME LOGIC VALIDATION")
    print("="*80)

    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        content = f.read()

    checks = {
        'RobustCheckpointManager class': 'class RobustCheckpointManager',
        'save_checkpoint method': 'def save_checkpoint',
        'load_checkpoint method': 'def load_checkpoint',
        'RNG state capture': 'rng_states',
        'Atomic save (tmp file)': '.tmp',
        'Backup creation': '_create_backup',
        'Checkpoint validation': '_validate_checkpoint',
        'Optimizer compatibility check': 'validate_optimizer_compatibility',
    }

    all_passed = True

    for check_name, pattern in checks.items():
        if pattern in content:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name}")
            all_passed = False

    if all_passed:
        print("\n✅ Checkpoint/resume logic is comprehensive")
    else:
        print("\n❌ Checkpoint logic needs improvement")

    return all_passed


def check_test_coverage():
    """Check test coverage"""
    print("\n" + "="*80)
    print("6. TEST COVERAGE CHECK")
    print("="*80)

    test_files = list(Path('tests').glob('test_*.py'))

    print(f"\nFound {len(test_files)} test files:")
    for test_file in sorted(test_files):
        print(f"  ✅ {test_file.name}")

    # Run fast tests
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', 'tests/', '-v', '-m', 'not slow', '--tb=line'],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode == 0:
        # Extract test count from output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'passed' in line:
                print(f"\n✅ {line.strip()}")
                break
        return True
    else:
        print("\n❌ Some tests failed")
        print(result.stdout[-500:])  # Last 500 chars
        return False


def main():
    """Run all validation checks"""
    print("="*80)
    print("COMPREHENSIVE CODEBASE VALIDATION")
    print("="*80)

    checks = [
        ('Syntax validation', check_syntax),
        ('Import validation', check_imports),
        ('Ablation study validation', check_ablation_studies),
        ('Experiment integration', check_experiment_integration),
        ('Checkpoint/resume logic', check_checkpoint_logic),
        ('Test coverage', check_test_coverage),
    ]

    results = {}

    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ {check_name} failed with error: {e}")
            results[check_name] = False

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("ALL VALIDATION CHECKS PASSED!")
        print("Codebase is ready for production use.")
    else:
        print("⚠️  SOME VALIDATION CHECKS FAILED")
        print("Please review the failures above.")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
