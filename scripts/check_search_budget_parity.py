#!/usr/bin/env python3
"""
Search Budget Parity Checker

Ensures equal hyperparameter search budgets across optimizers to prevent
strawman comparisons where one method gets unfairly more tuning trials.

This is HIGH-2 fix from the Research Validity Review.

Usage:
    python scripts/check_search_budget_parity.py
    python scripts/check_search_budget_parity.py --threshold 5.0
    
Author: GDSearch Remediation Team
Date: December 9, 2025
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np


def compute_grid_size(sweep_config):
    """Compute total grid size for a sweep configuration.
    
    Args:
        sweep_config: Dictionary containing hyperparameter arrays
        
    Returns:
        int: Total number of combinations
    """
    size = 1
    
    # Check for learning rate variations
    lr_keys = ['learning_rate', 'lr_values', 'lr']
    for key in lr_keys:
        if key in sweep_config and isinstance(sweep_config[key], list):
            size *= len(sweep_config[key])
            break
    
    # Check for weight decay
    wd_keys = ['weight_decay', 'weight_decay_values', 'wd']
    for key in wd_keys:
        if key in sweep_config and isinstance(sweep_config[key], list):
            size *= len(sweep_config[key])
            break
    
    # Check for momentum
    mom_keys = ['momentum', 'momentum_values']
    for key in mom_keys:
        if key in sweep_config and isinstance(sweep_config[key], list):
            size *= len(sweep_config[key])
            break
    
    # Check for betas
    if 'betas' in sweep_config and isinstance(sweep_config['betas'], list):
        size *= len(sweep_config['betas'])
    
    # Check for rho (RMSprop)
    if 'rho' in sweep_config and isinstance(sweep_config['rho'], list):
        size *= len(sweep_config['rho'])
    
    return max(size, 1)


def check_search_budget_parity(config_path, threshold=5.0):
    """Check if search budgets are balanced across optimizers.
    
    Args:
        config_path: Path to configuration file
        threshold: Maximum allowed ratio between largest and smallest grid
        
    Returns:
        dict: Analysis results
    """
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    
    grid_sizes = {}
    
    # Handle different config formats
    if 'sweeps' in config:
        sweeps = config['sweeps']
        
        # Array format (nn_tuning.json style)
        if isinstance(sweeps, list):
            for sweep in sweeps:
                if 'optimizer' in sweep:
                    opt_name = sweep['optimizer']
                    grid_sizes[opt_name] = compute_grid_size(sweep)
        
        # Object format (benchmark_hyperparameters.json style)
        elif isinstance(sweeps, dict):
            for opt_name, sweep_config in sweeps.items():
                grid_sizes[opt_name] = compute_grid_size(sweep_config)
    
    if not grid_sizes:
        return {
            'valid': True,
            'grid_sizes': {},
            'max_ratio': 1.0,
            'warning': 'No sweeps found in config'
        }
    
    max_size = max(grid_sizes.values())
    min_size = min(grid_sizes.values())
    max_ratio = max_size / min_size if min_size > 0 else float('inf')
    
    return {
        'valid': max_ratio <= threshold,
        'grid_sizes': grid_sizes,
        'max_ratio': max_ratio,
        'threshold': threshold,
        'max_optimizer': max(grid_sizes.items(), key=lambda x: x[1])[0],
        'min_optimizer': min(grid_sizes.items(), key=lambda x: x[1])[0]
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Check search budget parity across optimizers'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Maximum allowed grid size ratio (default: 5.0)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Specific config file to check (default: check all)'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    configs_dir = repo_root / 'configs'
    
    # Config files to check
    if args.config:
        config_files = [Path(args.config)]
    else:
        config_files = [
            configs_dir / 'nn_tuning.json',
            configs_dir / 'cifar10_tuning.json',
            configs_dir / 'benchmark_hyperparameters.json'
        ]
    
    print("=" * 80)
    print("SEARCH BUDGET PARITY CHECK")
    print("=" * 80)
    print(f"Threshold: {args.threshold}× (max/min grid size ratio)")
    print()
    
    all_valid = True
    results = []
    
    for config_file in config_files:
        if not config_file.exists():
            print(f"⚠  {config_file.name}: Not found (skipping)")
            continue
        
        print(f"{config_file.name}")
        print("-" * 80)
        
        result = check_search_budget_parity(config_file, args.threshold)
        
        if 'warning' in result:
            print(f"   {result['warning']}")
            print()
            continue
        
        # Display grid sizes
        for opt_name, size in sorted(result['grid_sizes'].items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"   {opt_name:20s}: {size:6d} combinations")
        
        print()
        print(f"   Max/Min Ratio: {result['max_ratio']:.2f}×")
        print(f"   Largest grid:  {result['max_optimizer']} ({max(result['grid_sizes'].values())} combos)")
        print(f"   Smallest grid: {result['min_optimizer']} ({min(result['grid_sizes'].values())} combos)")
        print()
        
        if result['valid']:
            print(f"   PASS: Ratio {result['max_ratio']:.2f}× ≤ {args.threshold}×")
        else:
            print(f"   FAIL: Ratio {result['max_ratio']:.2f}× > {args.threshold}×")
            print(f"   WARNING: Unequal search budgets create unfair comparisons!")
            print(f"   ⚠  Baselines may appear weaker due to under-tuning.")
            all_valid = False
        
        print()
        results.append((config_file.name, result))
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, r in results if r.get('valid', False))
    failed = sum(1 for _, r in results if not r.get('valid', True))
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)
    
    if not all_valid:
        print("\n❌ Search budget parity check FAILED")
        print("\nRecommendations:")
        print("1. Balance hyperparameter grids across optimizers")
        print("2. Ensure all methods get equal tuning opportunities")
        print("3. Document any intentional imbalances with justification")
        sys.exit(1)
    else:
        print("\n✅ Search budgets are balanced")
        sys.exit(0)


if __name__ == '__main__':
    main()
