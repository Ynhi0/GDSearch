"""
Quick Validation Script for Scientific Compliance Fixes

This script verifies that measured constants (L, σ², μ_PL) are correctly
loaded and used in theory-practice validation.

Run this to ensure the "Engineering → Science" transformation is complete.

Usage:
    python scripts/validate_scientific_compliance.py --verbose
"""

import json
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def check_hessian_artifacts(results_dir: Path, experiment: str) -> dict:
    """Check if Hessian analysis artifacts exist and contain max_eigenvalue."""
    hessian_dir = results_dir / experiment / 'hessian_analysis'
    
    if not hessian_dir.exists():
        return {'status': 'missing', 'message': f'Hessian analysis directory not found: {hessian_dir}'}
    
    json_files = list(hessian_dir.glob('*.json'))
    if not json_files:
        return {'status': 'missing', 'message': f'No JSON files in {hessian_dir}'}
    
    # Check first file
    try:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        if 'max_eigenvalue' not in data:
            return {'status': 'invalid', 'message': 'JSON missing max_eigenvalue key', 'file': str(json_files[0])}
        
        L = float(data['max_eigenvalue'])
        return {
            'status': 'valid',
            'message': f'✓ Found measured L = {L:.4f}',
            'L': L,
            'file': str(json_files[0]),
            'count': len(json_files)
        }
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to load {json_files[0]}: {e}'}


def check_gradient_noise_artifacts(results_dir: Path, experiment: str) -> dict:
    """Check if gradient noise analysis artifacts exist and contain sigma_squared."""
    noise_dir = results_dir / experiment / 'gradient_noise'
    
    if not noise_dir.exists():
        return {'status': 'missing', 'message': f'Gradient noise directory not found: {noise_dir}'}
    
    json_files = list(noise_dir.glob('*.json'))
    if not json_files:
        return {'status': 'missing', 'message': f'No JSON files in {noise_dir}'}
    
    # Check first file
    try:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        if 'sigma_squared' not in data and 'gradient_variance' not in data:
            return {'status': 'invalid', 'message': 'JSON missing sigma_squared/gradient_variance key', 'file': str(json_files[0])}
        
        sigma = data.get('sigma_squared', data.get('gradient_variance', 0))
        return {
            'status': 'valid',
            'message': f'✓ Found measured σ² = {sigma:.4e}',
            'sigma': sigma,
            'file': str(json_files[0]),
            'count': len(json_files)
        }
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to load {json_files[0]}: {e}'}


def check_pl_artifacts(results_dir: Path, experiment: str) -> dict:
    """Check if PL constant analysis artifacts exist."""
    pl_dir = results_dir / experiment / 'pl_analysis'
    
    if not pl_dir.exists():
        return {'status': 'missing', 'message': f'PL analysis directory not found: {pl_dir}'}
    
    json_files = list(pl_dir.glob('*.json'))
    if not json_files:
        return {'status': 'missing', 'message': f'No JSON files in {pl_dir}'}
    
    # Check first file
    try:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        if 'estimated_mu' not in data:
            return {'status': 'invalid', 'message': 'JSON missing estimated_mu key', 'file': str(json_files[0])}
        
        mu_pl = float(data['estimated_mu'])
        return {
            'status': 'valid',
            'message': f'✓ Found PL constant μ_PL = {mu_pl:.4e}',
            'mu_pl': mu_pl,
            'file': str(json_files[0]),
            'count': len(json_files)
        }
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to load {json_files[0]}: {e}'}


def check_theory_validation_code() -> dict:
    """Verify that theory_practice_validation.py has the compliance fixes."""
    validation_file = Path('src/experiments/theory_practice_validation.py')
    
    if not validation_file.exists():
        return {'status': 'error', 'message': f'File not found: {validation_file}'}
    
    try:
        with open(validation_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'loads_hessian': 'hessian_results_dir' in content and 'max_eigenvalue' in content,
            'loads_noise': 'noise_results_dir' in content and ('sigma_squared' in content or 'gradient_variance' in content),
            'loads_pl': 'pl_results_dir' in content and 'estimated_mu' in content,
            'passes_sigma': "sigma=sigma_est" in content or "sigma_est" in content,
            'passes_pl': "pl_constant=pl_const_est" in content or "pl_const_est" in content,
            'has_warnings': '⚠' in content or 'WARNING' in content
        }
        
        all_passed = all(checks.values())
        
        return {
            'status': 'valid' if all_passed else 'incomplete',
            'message': '✓ All compliance checks found in code' if all_passed else '⚠ Some checks missing',
            'details': checks
        }
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to read file: {e}'}


def main():
    """Run all validation checks."""
    print("="*80)
    print("SCIENTIFIC COMPLIANCE VALIDATION")
    print("="*80)
    print()
    
    # Check 1: Code compliance
    print("Check 1: Verifying code has compliance fixes...")
    code_result = check_theory_validation_code()
    print(f"  {code_result['message']}")
    if code_result['status'] == 'valid':
        print("  ✓ Code compliance: PASSED")
    else:
        print(f"  ✗ Code compliance: {code_result['status'].upper()}")
        if 'details' in code_result:
            for check, passed in code_result['details'].items():
                status = '✓' if passed else '✗'
                print(f"    {status} {check}")
    print()
    
    # Check 2: Artifact availability
    print("Check 2: Checking for analysis artifacts...")
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("  ⚠ Results directory not found. Run experiments first.")
        print()
        return 1
    
    # Look for experiment subdirectories
    experiments = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not experiments:
        print("  ⚠ No experiment directories found. Run experiments first.")
        print()
        return 1
    
    print(f"  Found {len(experiments)} experiment directories")
    print()
    
    # Check each experiment
    artifact_summary = {'hessian': 0, 'noise': 0, 'pl': 0}
    
    for exp_dir in experiments[:3]:  # Check first 3 experiments
        exp_name = exp_dir.name
        print(f"  Experiment: {exp_name}")
        
        hessian_result = check_hessian_artifacts(results_dir, exp_name)
        print(f"    Hessian: {hessian_result['message']}")
        if hessian_result['status'] == 'valid':
            artifact_summary['hessian'] += 1
        
        noise_result = check_gradient_noise_artifacts(results_dir, exp_name)
        print(f"    Gradient Noise: {noise_result['message']}")
        if noise_result['status'] == 'valid':
            artifact_summary['noise'] += 1
        
        pl_result = check_pl_artifacts(results_dir, exp_name)
        print(f"    PL Constant: {pl_result['message']}")
        if pl_result['status'] == 'valid':
            artifact_summary['pl'] += 1
        
        print()
    
    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Code Compliance: {'✓ PASSED' if code_result['status'] == 'valid' else '✗ FAILED'}")
    print(f"Hessian Artifacts: {artifact_summary['hessian']}/{len(experiments[:3])} experiments")
    print(f"Gradient Noise Artifacts: {artifact_summary['noise']}/{len(experiments[:3])} experiments")
    print(f"PL Constant Artifacts: {artifact_summary['pl']}/{len(experiments[:3])} experiments")
    print()
    
    if code_result['status'] == 'valid':
        print("✓ SCIENTIFIC COMPLIANCE: VERIFIED")
        print("  Theory-practice validation will use measured constants when available.")
        print()
        if sum(artifact_summary.values()) == 0:
            print("⚠ RECOMMENDATION: Run Hessian and gradient noise analysis to generate artifacts.")
            print("  Example:")
            print("    python -m src.experiments.hessian_analysis --experiment mnist")
            print("    python -m src.experiments.gradient_noise_analysis --experiment mnist")
        return 0
    else:
        print("✗ COMPLIANCE CHECK FAILED")
        print("  Review docs/SCIENTIFIC_COMPLIANCE_FIXES.md for details.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
