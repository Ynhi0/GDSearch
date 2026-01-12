#!/usr/bin/env python3
"""
Condition Number Sweep Experiment
Tests convergence rate scaling with condition number κ = L/μ

INTEGRATION FIX (Issue #4): Systematic validation of momentum's O(√κ) advantage
vs SGD's O(κ) convergence rate. This addresses the "Fixed Geometry Limitation"
from QA Report.

Usage:
    python run_condition_number_sweep.py --kappas 1,10,100,1000 --seeds 1,2,3
    python run_condition_number_sweep.py --quick  # Fast test with 3 κ values
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup reproducibility FIRST
from src.utils.reproducibility import setup_experiment_reproducibility
setup_experiment_reproducibility(seed=42, deterministic=False)

from src.analysis.condition_number_analysis import (
    quadratic_with_condition_number,
    sweep_condition_number_experiment,
    visualize_condition_number_sweep
)
from src.core.optimizers import SGD, SGDMomentum as Momentum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description='Condition Number Sweep - Validate momentum O(√κ) vs SGD O(κ)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full sweep (publication quality)
    python run_condition_number_sweep.py --kappas 1,5,10,50,100,500,1000 --seeds 1,2,3,4,5

    # Quick test (3 condition numbers)
    python run_condition_number_sweep.py --quick

    # Ultra-quick (1 seed, 3 κ values)
    python run_condition_number_sweep.py --ultra-quick
        """
    )

    parser.add_argument('--kappas', type=str, default='1,10,100,1000',
                       help='Comma-separated condition numbers (default: 1,10,100,1000)')
    parser.add_argument('--seeds', type=str, default='1,2,3',
                       help='Comma-separated random seeds (default: 1,2,3)')
    parser.add_argument('--dim', type=int, default=10,
                       help='Problem dimension (default: 10)')
    parser.add_argument('--max-iters', type=int, default=500,
                       help='Maximum iterations (default: 500)')
    parser.add_argument('--tol', type=float, default=1e-6,
                       help='Convergence tolerance (default: 1e-6)')
    parser.add_argument('--results-dir', type=str, default='results/condition_number_sweep',
                       help='Output directory (default: results/condition_number_sweep)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: κ ∈ {1,10,100}, 3 seeds')
    parser.add_argument('--ultra-quick', action='store_true',
                       help='Ultra-quick mode: κ ∈ {1,10,100}, 1 seed')

    args = parser.parse_args()

    # Parse condition numbers and seeds
    if args.ultra_quick:
        kappas = [1, 10, 100]
        seeds = [1]
        max_iters = 200
    elif args.quick:
        kappas = [1, 10, 100]
        seeds = [1, 2, 3]
        max_iters = 300
    else:
        kappas = [float(k) for k in args.kappas.split(',')]
        seeds = [int(s) for s in args.seeds.split(',')]
        max_iters = args.max_iters

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.info("="*70)
    logging.info("CONDITION NUMBER SWEEP EXPERIMENT")
    logging.info("="*70)
    logging.info(f"Condition numbers: {kappas}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Dimension: {args.dim}")
    logging.info(f"Max iterations: {max_iters}")
    logging.info(f"Tolerance: {args.tol}")
    logging.info(f"Results directory: {results_dir}")

    # Run sweep experiment
    results = []

    for kappa in kappas:
        logging.info(f"\n--- Testing κ = {kappa} ---")

        # Create quadratic function with this condition number
        obj_fn, grad_fn, metadata = quadratic_with_condition_number(
            kappa=kappa,
            n_dims=args.dim,
            random_rotation=True,  # Random rotation for non-axis-aligned geometry
            seed=42
        )

        for seed in seeds:
            setup_experiment_reproducibility(seed=seed)

            # Test SGD
            logging.info(f"  Seed {seed}: Testing SGD...")
            sgd = SGD(lr=0.1)  # Will need tuning per κ
            x0 = np.random.randn(args.dim) * 10  # Random initialization

            x_sgd = x0.copy()
            iters_sgd = 0
            for i in range(max_iters):
                grad = grad_fn(x_sgd)
                x_sgd = sgd.step(x_sgd, grad)
                iters_sgd += 1

                if obj_fn(x_sgd) < args.tol:
                    break

            # Test Momentum
            logging.info(f"  Seed {seed}: Testing Momentum...")
            momentum = Momentum(lr=0.1, beta=0.9)
            x_momentum = x0.copy()
            iters_momentum = 0
            for i in range(max_iters):
                grad = grad_fn(x_momentum)
                x_momentum = momentum.step(x_momentum, grad)
                iters_momentum += 1

                if obj_fn(x_momentum) < args.tol:
                    break

            # Record results
            results.append({
                'kappa': kappa,
                'seed': seed,
                'optimizer': 'SGD',
                'iterations': iters_sgd,
                'final_loss': obj_fn(x_sgd),
                'converged': obj_fn(x_sgd) < args.tol
            })

            results.append({
                'kappa': kappa,
                'seed': seed,
                'optimizer': 'Momentum',
                'iterations': iters_momentum,
                'final_loss': obj_fn(x_momentum),
                'converged': obj_fn(x_momentum) < args.tol
            })

            logging.info(f"    SGD: {iters_sgd} iters (loss={obj_fn(x_sgd):.2e})")
            logging.info(f"    Momentum: {iters_momentum} iters (loss={obj_fn(x_momentum):.2e})")

            # Speedup factor
            if iters_sgd > 0 and iters_momentum > 0:
                speedup = iters_sgd / iters_momentum
                logging.info(f"    Speedup: {speedup:.2f}x")

    # Save results
    df = pd.DataFrame(results)
    csv_path = results_dir / 'condition_number_sweep_results.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f"\n✓ Results saved to {csv_path}")

    # Compute summary statistics
    summary = df.groupby(['kappa', 'optimizer']).agg({
        'iterations': ['mean', 'std'],
        'converged': 'mean'
    }).reset_index()

    summary_path = results_dir / 'condition_number_sweep_summary.csv'
    summary.to_csv(summary_path, index=False)
    logging.info(f"✓ Summary saved to {summary_path}")

    # Visualize results
    logging.info("\nGenerating plots...")
    visualize_condition_number_sweep(df, output_path=str(results_dir / 'condition_number_sweep.png'))

    # Additional plot: Speedup vs κ
    fig, ax = plt.subplots(figsize=(10, 6))

    sgd_mean = df[df['optimizer'] == 'SGD'].groupby('kappa')['iterations'].mean()
    momentum_mean = df[df['optimizer'] == 'Momentum'].groupby('kappa')['iterations'].mean()
    speedup = sgd_mean / momentum_mean

    ax.plot(kappas, speedup.values, 'o-', linewidth=2, markersize=8, label='Observed Speedup')
    ax.plot(kappas, np.sqrt(kappas), '--', linewidth=2, label='Theoretical √κ', alpha=0.7)
    ax.set_xlabel('Condition Number κ', fontsize=12)
    ax.set_ylabel('Speedup (SGD iters / Momentum iters)', fontsize=12)
    ax.set_title('Momentum Speedup vs Condition Number', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    speedup_path = results_dir / 'momentum_speedup_vs_kappa.png'
    plt.savefig(speedup_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✓ Speedup plot saved to {speedup_path}")

    # Print final summary
    logging.info("\n" + "="*70)
    logging.info("FINAL SUMMARY")
    logging.info("="*70)
    logging.info(f"Tested {len(kappas)} condition numbers × {len(seeds)} seeds")
    logging.info(f"Total experiments: {len(df)} ({len(df)//2} SGD + {len(df)//2} Momentum)")
    logging.info("Mean iterations to convergence (averaged over seeds):")
    for kappa in kappas:
        sgd_iters = df[(df['kappa'] == kappa) & (df['optimizer'] == 'SGD')]['iterations'].mean()
        mom_iters = df[(df['kappa'] == kappa) & (df['optimizer'] == 'Momentum')]['iterations'].mean()
        speedup_val = sgd_iters / mom_iters if mom_iters > 0 else 0
        logging.info(f"  κ = {kappa:6.0f}: SGD = {sgd_iters:6.1f}, Momentum = {mom_iters:6.1f}, Speedup = {speedup_val:.2f}x")

    logging.info("✓ Condition number sweep complete!")
    logging.info(f"✓ Results: {results_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
