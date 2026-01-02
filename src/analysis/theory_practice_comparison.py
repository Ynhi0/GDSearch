"""
Theory vs Practice Convergence Comparison
Compare observed convergence rates with theoretical predictions.

This module implements comparison between:
1. Actual training convergence (from CSV results)
2. Theoretical convergence bounds (from optimization theory)

Required by research proposal:
"đối chiếu tốc độ hội tụ quan sát được với các dự đoán lý thuyết"
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import curve_fit


def theoretical_sgd_convex(iterations: np.ndarray, L: float, R: float, 
                           sigma: float = 0.0, alpha: Optional[float] = None) -> np.ndarray:
    """
    Theoretical convergence rate for SGD on convex functions.
    
    Theory: E[f(x_k) - f*] ≤ O(1/√k) for constant step size
            or O(1/k) for decreasing step size α_k = C/k
    
    Args:
        iterations: Array of iteration numbers
        L: Lipschitz constant of gradient
        R: Initial distance to optimum
        sigma: Gradient noise level
        alpha: Step size (if None, uses decreasing α = 1/√k)
        
    Returns:
        Array of theoretical suboptimality bounds
    """
    k = iterations + 1  # Avoid k=0
    
    if alpha is None:
        # Decreasing step size: α_k = 1/√k
        # Bound: E[f(x_k) - f*] ≤ (L*R²)/√k + σ²/(2√k)
        bound = (L * R**2) / np.sqrt(k) + (sigma**2) / (2 * np.sqrt(k))
    else:
        # Constant step size
        # Bound: E[f(x_k) - f*] ≤ (L*R²)/(α*k) + α*σ²/2
        bound = (L * R**2) / (alpha * k) + (alpha * sigma**2) / 2
    
    return bound


def theoretical_sgd_strongly_convex(iterations: np.ndarray, mu: float, L: float,
                                    initial_subopt: float = 1.0,
                                    alpha: Optional[float] = None) -> np.ndarray:
    """
    Theoretical convergence rate for SGD on strongly convex functions.
    
    Theory: E[||x_k - x*||²] ≤ (1 - μα)^k * ||x_0 - x*||²  (geometric convergence)
    
    Args:
        iterations: Array of iteration numbers
        mu: Strong convexity parameter
        L: Lipschitz constant
        initial_subopt: Initial suboptimality
        alpha: Step size (if None, uses α = 1/L)
        
    Returns:
        Array of theoretical error bounds
    """
    k = iterations
    
    if alpha is None:
        alpha = 1.0 / L
    
    # Geometric convergence rate
    rho = 1 - mu * alpha
    bound = initial_subopt * (rho ** k)
    
    return bound


def theoretical_pl_condition(iterations: np.ndarray, mu: float, L: float,
                            initial_loss: float) -> np.ndarray:
    """
    Theoretical convergence under Polyak-Łojasiewicz (PL) condition.
    
    Theory: f(x_k) - f* ≤ (1 - μ/L)^k * (f(x_0) - f*)  (linear convergence)
    
    Args:
        iterations: Array of iteration numbers
        mu: PL constant
        L: Lipschitz constant
        initial_loss: f(x_0) - f*
        
    Returns:
        Array of theoretical loss bounds
    """
    k = iterations
    
    # Linear convergence rate
    rho = 1 - mu / L
    bound = initial_loss * (rho ** k)
    
    return bound


def theoretical_momentum(iterations: np.ndarray, L: float, mu: float,
                        initial_error: float, beta: float = 0.9) -> np.ndarray:
    """
    Theoretical convergence rate for momentum method on strongly convex functions.
    
    Theory: ||x_k - x*|| ≤ O((1 - √(μ/L))^k)  with optimal β
    
    Args:
        iterations: Array of iteration numbers
        L: Lipschitz constant
        mu: Strong convexity parameter
        initial_error: ||x_0 - x*||
        beta: Momentum coefficient
        
    Returns:
        Array of theoretical error bounds
    """
    k = iterations
    
    # Condition number
    kappa = L / mu
    
    # Optimal convergence rate with momentum
    if beta == 'optimal':
        # β_optimal = (√κ - 1) / (√κ + 1)
        sqrt_kappa = np.sqrt(kappa)
        beta_opt = (sqrt_kappa - 1) / (sqrt_kappa + 1)
        rho = ((sqrt_kappa - 1) / (sqrt_kappa + 1))
    else:
        # Approximate rate for given β
        rho = max(beta, 1 - 1/np.sqrt(kappa))
    
    bound = initial_error * (rho ** k)
    
    return bound


def theoretical_adam(iterations: np.ndarray, L: float, 
                    initial_loss: float, alpha: float = 0.001,
                    beta1: float = 0.9, beta2: float = 0.999) -> np.ndarray:
    """
    Theoretical convergence rate for Adam (under convexity assumptions).
    
    Theory: Regret bound R_T ≤ O(√T) for convex functions
            Implies f(x_k) - f* ≤ O(1/√k) on average
    
    Args:
        iterations: Array of iteration numbers
        L: Lipschitz constant
        initial_loss: f(x_0) - f*
        alpha, beta1, beta2: Adam hyperparameters
        
    Returns:
        Array of theoretical loss bounds
    """
    k = iterations + 1
    
    # Simplified bound: O(1/√k) for convex case
    # More complex bounds exist but require additional constants
    bound = initial_loss / np.sqrt(k)
    
    return bound


def fit_empirical_rate(iterations: np.ndarray, values: np.ndarray,
                      rate_type: str = 'linear') -> Tuple[float, np.ndarray, float]:
    """
    Fit empirical convergence rate to observed data.
    
    Args:
        iterations: Iteration numbers
        values: Observed values (loss or error)
        rate_type: 'linear' (geometric), 'sublinear' (O(1/k)), or 'sqrt' (O(1/√k))
        
    Returns:
        Tuple of (fitted_rate, fitted_curve, R²)
    """
    # Remove zeros and infinities
    valid = (values > 0) & np.isfinite(values) & (iterations > 0)
    iters_valid = iterations[valid]
    vals_valid = values[valid]
    
    if len(vals_valid) < 3:
        return 0.0, np.zeros_like(iterations), 0.0
    
    try:
        if rate_type == 'linear':
            # Geometric convergence: f(k) = C * exp(-λk) or C * ρ^k
            def func_linear(k, C, rho):
                return C * (rho ** k)
            
            # Initial guess
            p0 = [vals_valid[0], 0.95]
            bounds = ([0, 0], [vals_valid[0] * 10, 0.9999])
            
            params, _ = curve_fit(func_linear, iters_valid, vals_valid, p0=p0, bounds=bounds, maxfev=5000)
            fitted = func_linear(iterations, *params)
            fitted_func = func_linear
            rate = -np.log(params[1])  # λ = -log(ρ)
            
        elif rate_type == 'sublinear':
            # Sublinear convergence: f(k) = C / k
            def func_sublinear(k, C):
                return C / k
            
            p0 = [vals_valid[0] * iters_valid[0]]
            params, _ = curve_fit(func_sublinear, iters_valid, vals_valid, p0=p0, maxfev=5000)
            fitted = func_sublinear(iterations, *params)
            fitted_func = func_sublinear
            rate = params[0]  # C constant
            
        elif rate_type == 'sqrt':
            # Square root convergence: f(k) = C / √k
            def func_sqrt(k, C):
                return C / np.sqrt(k)
            
            p0 = [vals_valid[0] * np.sqrt(iters_valid[0])]
            params, _ = curve_fit(func_sqrt, iters_valid, vals_valid, p0=p0, maxfev=5000)
            fitted = func_sqrt(iterations, *params)
            fitted_func = func_sqrt
            rate = params[0]
            
        else:
            raise ValueError(f"Unknown rate type: {rate_type}")
        
        # Compute R²
        ss_res = np.sum((vals_valid - fitted_func(iters_valid, *params))**2)
        ss_tot = np.sum((vals_valid - np.mean(vals_valid))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return rate, fitted, r_squared
        
    except Exception as e:
        logging.info(f"Warning: Curve fitting failed: {e}")
        return 0.0, np.zeros_like(iterations), 0.0


def compare_theory_practice(training_csv: str, optimizer_name: str,
                           output_dir: str, 
                           L: float = 1.0, mu: Optional[float] = None,
                           **kwargs) -> Dict:
    """
    Compare actual training convergence with theoretical predictions.
    
    Args:
        training_csv: Path to training results CSV (must have 'iteration' and 'loss' columns)
        optimizer_name: Name of optimizer for plot titles
        output_dir: Directory to save comparison plots
        L: Estimated Lipschitz constant
        mu: Strong convexity parameter (if applicable)
        **kwargs: Additional parameters for theoretical bounds
        
    Returns:
        dict: Comparison statistics
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load training data
    df = pd.read_csv(training_csv)
    
    # Ensure required columns exist
    if 'iteration' not in df.columns or 'loss' not in df.columns:
        logging.info(f"Warning: CSV must have 'iteration' and 'loss' columns")
        return {}
    
    iterations = np.asarray(df['iteration'].values)
    losses = np.asarray(df['loss'].values, dtype=float)

    # Filter valid data
    valid = (losses > 0) & np.isfinite(losses)
    iterations = iterations[valid]
    losses = losses[valid]
    
    if len(losses) < 10:
        logging.info("Warning: Not enough data points for comparison")
        return {}
    
    # Fit empirical rate
    empirical_rate, fitted_curve, r_squared = fit_empirical_rate(
        iterations, losses, rate_type='linear' if mu else 'sublinear'
    )
    
    # Compute theoretical bound
    initial_loss = losses[0]
    
    if mu is not None:
        # Strongly convex case
        if 'PL' in optimizer_name.upper() or 'pl' in kwargs:
            theoretical = theoretical_pl_condition(iterations, mu, L, initial_loss)
        else:
            theoretical = theoretical_sgd_strongly_convex(iterations, mu, L, initial_loss)
        theory_label = f"Theory: O((1-μ/L)^k), μ/L={mu/L:.4f}"
        
        # Compute theoretical convergence rate: ρ = 1 - μ/L
        theoretical_rate = 1 - mu / L
    else:
        # General non-convex case
        R = kwargs.get('R', 1.0)
        theoretical = theoretical_sgd_convex(iterations, L, R)
        theory_label = f"Theory: O(1/√k), L={L}"
        
        # Theoretical rate is sublinear (no exponential rate)
        theoretical_rate = None
    
    # Compute Optimality Gap
    # This is the TRUE theory-practice comparison: difference between fitted and predicted rate
    # Without this, curve fitting is just decoration, not science
    optimality_gap = None
    optimality_gap_pct = None
    
    if mu is not None and empirical_rate > 0 and theoretical_rate is not None:
        # For geometric convergence: compare -log(ρ) values
        # Empirical rate from fit is already -log(ρ_empirical)
        theoretical_rate_log = -np.log(theoretical_rate)
        
        # Optimality gap: how far is empirical rate from theoretical rate?
        optimality_gap = abs(empirical_rate - theoretical_rate_log)
        optimality_gap_pct = (optimality_gap / theoretical_rate_log) * 100
        
        logging.info(f"  Theoretical convergence rate (ρ): {theoretical_rate:.6f}")
        logging.info(f"  Theoretical rate (-log ρ): {theoretical_rate_log:.6f}")
        logging.info(f"  Empirical rate (-log ρ_fit): {empirical_rate:.6f}")
        logging.info(f"  ⚠️  OPTIMALITY GAP: {optimality_gap:.6f} ({optimality_gap_pct:.2f}%)")
        
        if optimality_gap_pct > 10:
            logging.info(f"  WARNING: Large optimality gap (>10%) indicates theory-practice mismatch")
        elif optimality_gap_pct < 5:
            logging.info(f"  ✓ Good agreement: optimality gap <5%")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{optimizer_name} - Theory vs Practice Comparison', fontsize=14, fontweight='bold')
    
    # Linear scale plot
    ax1.plot(iterations, losses, 'b-', alpha=0.6, linewidth=2, label='Actual Training')
    ax1.plot(iterations, fitted_curve, 'g--', linewidth=2, label=f'Empirical Fit (R²={r_squared:.3f})')
    ax1.plot(iterations, theoretical, 'r-.', linewidth=2, label=theory_label)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Linear Scale')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Log scale plot (shows rate more clearly)
    ax2.semilogy(iterations, losses, 'b-', alpha=0.6, linewidth=2, label='Actual Training')
    ax2.semilogy(iterations, fitted_curve, 'g--', linewidth=2, label=f'Empirical Fit')
    ax2.semilogy(iterations, theoretical, 'r-.', linewidth=2, label=theory_label)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Logarithmic Scale')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / f'{optimizer_name}_theory_vs_practice.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"✓ Theory-practice comparison saved to {plot_path}")
    
    # Compute deviation statistics
    # Normalize by theoretical values to get relative error
    relative_errors = np.abs(losses - theoretical) / (np.abs(theoretical) + 1e-10)
    
    stats = {
        'optimizer': optimizer_name,
        'empirical_rate': float(empirical_rate),
        'theoretical_rate': float(theoretical_rate) if theoretical_rate is not None else None,
        'optimality_gap': float(optimality_gap) if optimality_gap is not None else None,
        'optimality_gap_pct': float(optimality_gap_pct) if optimality_gap_pct is not None else None,
        'fit_r_squared': float(r_squared),
        'mean_relative_error': float(np.mean(relative_errors)),
        'median_relative_error': float(np.median(relative_errors)),
        'max_relative_error': float(np.max(relative_errors)),
        'final_loss': float(losses[-1]),
        'theoretical_final': float(theoretical[-1])
    }
    
    # Save stats to CSV in output_dir
    stats_csv = Path(output_dir) / f'{optimizer_name}_theory_practice_stats.csv'
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)
    logging.info(f"✓ Stats saved to {stats_csv}")
    
    return stats


def batch_compare_optimizers(results_dir: str, output_dir: str,
                             optimizers: list, L: float = 1.0,
                             mu: Optional[float] = None):
    """
    Compare theory vs practice for multiple optimizers.
    
    Args:
        results_dir: Directory containing training CSVs
        output_dir: Directory to save comparison plots
        optimizers: List of optimizer names
        L: Lipschitz constant estimate
        mu: Strong convexity parameter (if applicable)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    
    for opt in optimizers:
        # Find CSV file for this optimizer
        csv_pattern = f"*{opt}*.csv"
        csv_files = list(Path(results_dir).glob(csv_pattern))
        
        if len(csv_files) == 0:
            logging.info(f"Warning: No CSV found for {opt}")
            continue
        
        # Use first matching file
        csv_file = csv_files[0]
        
        stats = compare_theory_practice(
            str(csv_file), opt, output_dir, L=L, mu=mu
        )
        
        if stats:
            all_stats.append(stats)
    
    # Initialize comparison_df to ensure it's bound in all code paths
    comparison_df = None
    
    # Create comparison table
    if all_stats:
        comparison_df = pd.DataFrame(all_stats)
        from typing import cast
        comparison_df = cast(pd.DataFrame, comparison_df).sort_values(by=['fit_r_squared'], ascending=False)
        
        table_path = Path(output_dir) / 'theory_practice_comparison.csv'
        comparison_df.to_csv(table_path, index=False)
        
        logging.info(f"\n✓ Comparison table saved to {table_path}")
        logging.info("\nTheory-Practice Fit Quality (R²):")
        print(comparison_df[['optimizer', 'fit_r_squared', 'mean_relative_error']])
    
    return comparison_df if all_stats else None


def fit_convergence_rate(iterations: np.ndarray, values: np.ndarray) -> Dict:
    """
    Fit convergence rate with automatic model selection.
    
    Tries multiple convergence models (linear/geometric, sublinear O(1/k), sqrt O(1/√k))
    and returns the best fit based on R² score.
    
    Args:
        iterations: Array of iteration numbers
        values: Observed values (typically gradient norms or losses)
        
    Returns:
        Dict containing:
            - 'best_model': Name of best-fitting model ('linear', 'sublinear', or 'sqrt')
            - 'best_r_squared': R² score of best model
            - 'linear': Dict with 'rate', 'fitted_curve', 'r_squared', 'formula'
            - 'sublinear': Dict with 'alpha', 'C', 'fitted_curve', 'r_squared', 'formula'
            - 'sqrt': Dict with 'C', 'fitted_curve', 'r_squared', 'formula'
    """
    # Try all three models
    models = {}
    
    # Linear/Geometric convergence: f(k) = C * ρ^k
    rate_linear, fitted_linear, r2_linear = fit_empirical_rate(iterations, values, 'linear')
    models['linear'] = {
        'rate': rate_linear,
        'fitted_curve': fitted_linear,
        'r_squared': r2_linear,
        'formula': f'{values[0]:.4e} * exp(-{rate_linear:.4f} * k)' if len(values) > 0 else 'N/A'
    }
    
    # Sublinear convergence: f(k) = C / k
    rate_sublinear, fitted_sublinear, r2_sublinear = fit_empirical_rate(iterations, values, 'sublinear')
    models['sublinear'] = {
        'C': rate_sublinear,
        'alpha': 1.0,  # O(1/k^1)
        'fitted_curve': fitted_sublinear,
        'r_squared': r2_sublinear,
        'formula': f'{rate_sublinear:.4e} / k'
    }
    
    # Square root convergence: f(k) = C / √k
    rate_sqrt, fitted_sqrt, r2_sqrt = fit_empirical_rate(iterations, values, 'sqrt')
    models['sqrt'] = {
        'C': rate_sqrt,
        'fitted_curve': fitted_sqrt,
        'r_squared': r2_sqrt,
        'formula': f'{rate_sqrt:.4e} / sqrt(k)'
    }
    
    # Select best model by R²
    best_model = 'linear'
    best_r2 = r2_linear
    
    if r2_sublinear > best_r2:
        best_model = 'sublinear'
        best_r2 = r2_sublinear
    
    if r2_sqrt > best_r2:
        best_model = 'sqrt'
        best_r2 = r2_sqrt
    
    return {
        'best_model': best_model,
        'best_r_squared': best_r2,
        'linear': models['linear'],
        'sublinear': models['sublinear'],
        'sqrt': models['sqrt']
    }


if __name__ == "__main__":
    logging.info("Theory vs Practice Comparison Module")
    print("=" * 60)
    logging.info("This module compares observed convergence with theoretical bounds.")
    logging.info("See experiments for integration examples.")
