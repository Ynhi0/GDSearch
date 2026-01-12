"""
Convergence Rate Theory vs Practice Validation

Validates theoretical convergence rate guarantees (O(1/k)) against empirical observations.
Addresses research proposal requirement for "đối chiếu tốc độ hội tụ quan sát được
với các dự đoán lý thuyết".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from src.analysis.theory_practice_comparison import fit_convergence_rate

try:
    from src.core.test_functions import Rosenbrock, IllConditionedQuadratic
    from src.core.optimizers import SGD, SGDMomentum, Adam, RMSProp
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.test_functions import Rosenbrock, IllConditionedQuadratic
    from src.core.optimizers import SGD, SGDMomentum, Adam, RMSProp

def validate_convergence_rate(
    optimizer_name: str,
    optimizer_class,
    optimizer_params: Dict,
    test_function: str = 'ill_conditioned',
    x0: np.ndarray = np.array([-1.5, 2.0]),
    max_iters: int = 2000,
    tol: float = 1e-8,
    noise_std: float = 0.0  # GAP FIX: Add noise_std parameter for true SGD
):
    """
    Validate convergence rate for a single optimizer.

    GAP FIX: Added noise_std parameter. Without noise, "SGD" is actually deterministic GD.
    To validate true SGD convergence (O(1/√k)), set noise_std > 0.

    Args:
        optimizer_name: Name for results
        optimizer_class: Optimizer class to test
        optimizer_params: Parameters for optimizer initialization
        test_function: Which test function to use
        x0: Initial point
        max_iters: Maximum iterations
        tol: Tolerance for convergence
        noise_std: Gradient noise std dev (0 = deterministic GD, >0 = SGD simulation)

    Returns:
        results: Dictionary with trajectory and fitted rates
    """
    # Select test function
    if test_function == 'ill_conditioned':
        test_fn = IllConditionedQuadratic()
    else:
        test_fn = Rosenbrock()

    func = test_fn.compute
    # GAP FIX: Pass noise_std to gradient for true SGD simulation
    # Without noise, SGD experiments are actually GD (deterministic)
    grad_func = lambda x, y: np.array(test_fn.gradient(x, y, noise_std=noise_std))

    # Initialize optimizer
    optimizer = optimizer_class(**optimizer_params)

    # Run optimization
    params = np.array(x0, dtype=float)
    trajectory = [params.copy()]
    losses = [func(*params)]
    grad_norms = []

    for iter_num in range(max_iters):
        grad = grad_func(*params)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)

        if grad_norm < tol:
            logging.info(f"{optimizer_name}: Converged at iteration {iter_num}")
            break

        params = optimizer.step(params, grad)
        trajectory.append(params.copy())
        losses.append(func(*params))

    # Fit convergence rate
    iterations = np.arange(1, len(grad_norms) + 1)
    grad_norms_arr = np.array(grad_norms)

    fit_results = fit_convergence_rate(iterations, grad_norms_arr)

    return {
        'optimizer': optimizer_name,
        'trajectory': np.array(trajectory),
        'losses': np.array(losses),
        'grad_norms': grad_norms_arr,
        'iterations': iterations,
        'fit_results': fit_results,
        'convergence_iters': len(grad_norms)
    }


def run_convergence_rate_comparison(
    output_dir: str = 'results/convergence_rate_validation'
):
    """
    Compare theoretical vs empirical convergence rates for multiple optimizers.

    Theory predictions (for L-smooth non-convex functions):
    - GD: O(1/k) for gradient norm squared
    - SGD with momentum: O(1/k) to O(1/sqrt(k)) depending on analysis
    - Adam: O(1/sqrt(k)) in some analyses, O(1/k) in others

    This experiment validates these predictions empirically.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Test on ill-conditioned quadratic (L-smooth, non-convex in general sense)
    x0 = np.array([-1.5, 2.0])

    optimizers_config = [
        ('SGD', SGD, {'lr': 0.01}),
        ('SGD_Momentum_0.5', SGDMomentum, {'lr': 0.01, 'beta': 0.5}),
        ('SGD_Momentum_0.9', SGDMomentum, {'lr': 0.01, 'beta': 0.9}),
        ('Adam', Adam, {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999}),
        ('RMSProp', RMSProp, {'lr': 0.01, 'beta': 0.9})
    ]

    all_results = []

    print("="*60)
    print("Convergence Rate Validation: Theory vs Practice")
    print("="*60)

    for opt_name, opt_class, opt_params in optimizers_config:
        print(f"\nTesting {opt_name}...")

        result = validate_convergence_rate(
            opt_name, opt_class, opt_params,
            test_function='ill_conditioned',
            x0=x0,
            max_iters=2000
        )

        fit_res = result['fit_results']
        best_model = fit_res.get('best_model', 'unknown')
        best_r2 = fit_res.get('best_r_squared', 0.0)

        if best_model == 'sublinear':
            alpha = fit_res['sublinear'].get('alpha', 0.0)
            formula = fit_res['sublinear'].get('formula', 'N/A')
            print(f"  Best fit: Sublinear O(1/k^{alpha:.2f})")
            print(f"  Formula: {formula}")
            print(f"  R²: {best_r2:.4f}")
        elif best_model == 'linear':
            rate = fit_res['linear'].get('rate', 0.0)
            formula = fit_res['linear'].get('formula', 'N/A')
            print(f"  Best fit: Linear (geometric) {rate:.4f}^k")
            print(f"  Formula: {formula}")
            print(f"  R²: {best_r2:.4f}")

        all_results.append(result)

        # Save individual result
        df = pd.DataFrame({
            'iteration': result['iterations'],
            'grad_norm': result['grad_norms'],
            'loss': result['losses'][:-1]  # Losses are 1 longer
        })
        df.to_csv(Path(output_dir) / f'{opt_name}_trajectory.csv', index=False)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Gradient norm vs iteration (log-log)
    ax1 = axes[0]
    for result in all_results:
        ax1.loglog(result['iterations'], result['grad_norms'],
                  label=result['optimizer'], alpha=0.7, linewidth=2)

    # Add theoretical reference lines
    k_ref = np.logspace(0, 3, 100)
    ax1.loglog(k_ref, 1.0 / k_ref, 'k--', alpha=0.5, label='O(1/k) reference', linewidth=1)
    ax1.loglog(k_ref, 1.0 / np.sqrt(k_ref), 'k:', alpha=0.5, label='O(1/√k) reference', linewidth=1)

    ax1.set_xlabel('Iteration k', fontsize=12)
    ax1.set_ylabel('||∇f(x_k)||', fontsize=12)
    ax1.set_title('Convergence Rate: Theory vs Practice', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Loss vs iteration (log scale)
    ax2 = axes[1]
    for result in all_results:
        ax2.semilogy(result['iterations'], result['losses'][:-1],
                    label=result['optimizer'], alpha=0.7, linewidth=2)

    ax2.set_xlabel('Iteration k', fontsize=12)
    ax2.set_ylabel('f(x_k)', fontsize=12)
    ax2.set_title('Loss Convergence', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'convergence_rate_comparison.png', dpi=150)
    plt.close()

    # Create summary table
    summary = []
    for result in all_results:
        fit_res = result['fit_results']
        row = {
            'Optimizer': result['optimizer'],
            'Convergence_Iters': result['convergence_iters'],
            'Best_Model': fit_res.get('best_model', 'unknown'),
            'R_Squared': fit_res.get('best_r_squared', 0.0)
        }

        if fit_res.get('best_model') == 'sublinear':
            row['Alpha'] = fit_res['sublinear'].get('alpha', np.nan)
            row['Theoretical_Match'] = 'O(1/k)' if 0.8 <= row['Alpha'] <= 1.2 else 'Deviates'
        else:
            row['Convergence_Rate'] = fit_res.get('linear', {}).get('rate', np.nan)
            row['Theoretical_Match'] = 'Linear (fast)' if row['Convergence_Rate'] < 0.95 else 'Slower'

        summary.append(row)

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(Path(output_dir) / 'convergence_rate_summary.csv', index=False)

    print("\n" + "="*60)
    print("Summary of Convergence Rate Analysis:")
    print("="*60)
    print(df_summary.to_string(index=False))

    print(f"\nResults saved to {output_dir}/")

    return all_results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    results = run_convergence_rate_comparison()

    print("\nTheory vs Practice Validation Complete!")
    print("   Check plots and CSV files for detailed analysis.")
