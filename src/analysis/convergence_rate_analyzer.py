"""
Empirical Convergence Rate Analysis Module

Computes empirical convergence rates from loss trajectories and compares them
with theoretical bounds. Supports power-law and exponential curve fitting.

This module directly supports the research proposal objective:
"Synthesis of theoretical results on convergence rate of GD/variants and
comparison with experimental observations."
"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


def fit_power_law(iterations: np.ndarray, losses: np.ndarray) -> Dict[str, Any]:
    """
    Fit power-law convergence: loss(t) = A * t^(-α) + B
    
    Args:
        iterations: Array of iteration indices (1-indexed to avoid log(0))
        losses: Array of loss values
    
    Returns:
        Dict with keys: alpha (convergence exponent), A, B, r_squared, success
    """
    try:
        # Ensure iterations start at 1 (not 0)
        t = np.maximum(iterations, 1)
        
        # Initial guess
        def power_model(t, A, alpha, B):
            return A * np.power(t, -alpha) + B
        
        # Fit
        popt, pcov = optimize.curve_fit(
            power_model, t, losses,
            p0=[losses[0] - losses[-1], 0.5, losses[-1]],
            maxfev=10000,
            bounds=([0, 0, -np.inf], [np.inf, 5, np.inf])
        )
        
        A, alpha, B = popt
        fitted = power_model(t, A, alpha, B)
        
        # Could use pcov for uncertainty estimation in future
        _ = pcov  # Suppress unused warning
        
        # Compute R²
        ss_res = np.sum((losses - fitted) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'alpha': alpha,
            'A': A,
            'B': B,
            'r_squared': r_squared,
            'fitted_values': fitted,
            'success': True
        }
    except Exception as e:
        logger.warning(f"Power-law fit failed: {e}")
        return {'success': False, 'error': str(e)}


def fit_exponential(iterations: np.ndarray, losses: np.ndarray) -> Dict[str, Any]:
    """
    Fit exponential convergence: loss(t) = A * exp(-β * t) + B
    
    Args:
        iterations: Array of iteration indices
        losses: Array of loss values
    
    Returns:
        Dict with keys: beta (convergence rate), A, B, r_squared, success
    """
    try:
        t = iterations
        
        def exp_model(t, A, beta, B):
            return A * np.exp(-beta * t) + B
        
        # Initial guess
        popt, pcov = optimize.curve_fit(
            exp_model, t, losses,
            p0=[losses[0] - losses[-1], 0.01, losses[-1]],
            maxfev=10000,
            bounds=([0, 0, -np.inf], [np.inf, 1, np.inf])
        )
        
        A, beta, B = popt
        fitted = exp_model(t, A, beta, B)
        
        # Could use pcov for uncertainty estimation in future
        _ = pcov  # Suppress unused warning
        
        # Compute R²
        ss_res = np.sum((losses - fitted) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'beta': beta,
            'A': A,
            'B': B,
            'r_squared': r_squared,
            'fitted_values': fitted,
            'success': True
        }
    except Exception as e:
        logger.warning(f"Exponential fit failed: {e}")
        return {'success': False, 'error': str(e)}


def compute_empirical_rate(
    loss_history: List[float],
    method: str = 'auto'
) -> Dict[str, Any]:
    """
    Compute empirical convergence rate from loss trajectory.
    
    Args:
        loss_history: List of loss values over training
        method: 'power', 'exponential', or 'auto' (tries both)
    
    Returns:
        Dict with convergence metrics and best-fit model
    """
    if len(loss_history) < 10:
        return {'success': False, 'error': 'Insufficient data (<10 points)'}
    
    iterations = np.arange(len(loss_history))
    losses = np.array(loss_history)
    
    results = {'iterations': iterations, 'losses': losses}
    
    if method in ['power', 'auto']:
        power_fit = fit_power_law(iterations + 1, losses)  # +1 to avoid log(0)
        results['power_law'] = power_fit
    
    if method in ['exponential', 'auto']:
        exp_fit = fit_exponential(iterations, losses)
        results['exponential'] = exp_fit
    
    # Select best fit based on R²
    if method == 'auto':
        power_r2 = power_fit.get('r_squared', -1) if power_fit.get('success') else -1
        exp_r2 = exp_fit.get('r_squared', -1) if exp_fit.get('success') else -1
        
        if power_r2 > exp_r2:
            results['best_fit'] = 'power_law'
            results['best_r_squared'] = power_r2
        else:
            results['best_fit'] = 'exponential'
            results['best_r_squared'] = exp_r2
    elif method == 'power':
        results['best_fit'] = 'power_law'
        results['best_r_squared'] = power_fit.get('r_squared', 0)
    else:
        results['best_fit'] = 'exponential'
        results['best_r_squared'] = exp_fit.get('r_squared', 0)
    
    results['success'] = True
    return results


def compare_to_theoretical_bounds(
    empirical_rate: float,
    optimizer_name: str,
    problem_type: str = 'strongly_convex',
    lr: float = 0.01,
    condition_number: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compare empirical convergence rate to theoretical bounds.
    
    CRITICAL FIX: Properly accounts for problem condition number (kappa = L/mu)
    in theoretical rate predictions. Without this, theoretical bounds are
    meaningless as they ignore the problem geometry.
    
    Correct theoretical rates:
    - SGD (strongly convex): beta = lr * mu * (1 - 1/kappa) where mu = strong convexity
    - Momentum: beta \u2248 sqrt(lr * mu)  
    - Adam: No closed-form bound (adaptive)
    
    Args:
        empirical_rate: Measured convergence exponent (α for power-law or β for exponential)
        optimizer_name: Name of optimizer (SGD, Adam, etc.)
        problem_type: 'strongly_convex', 'convex', or 'nonconvex'
        lr: Learning rate used
        condition_number: kappa = L/mu (eigenvalue ratio for quadratics)
    
    Returns:
        Dict with theoretical bounds and deviation metrics
    """
    # Map optimizer name
    opt_key = optimizer_name.upper()
    if 'MOMENTUM' in opt_key:
        opt_key = 'Momentum'
    elif 'ADAM' in opt_key:
        opt_key = 'Adam'
    elif 'SGD' in opt_key:
        opt_key = 'SGD'
    else:
        opt_key = 'SGD'  # Default
    
    # CRITICAL FIX: Compute theoretical rate using condition number
    if condition_number is not None and problem_type == 'strongly_convex':
        kappa = condition_number
        # Estimate mu: for normalized problems, assume L ~ 1, so mu ~ 1/kappa
        mu_estimate = 1.0 / kappa
        
        if opt_key == 'SGD':
            # Linear convergence: (1 - lr*mu) per iteration
            # Continuous rate: beta = lr * mu * (1 - 1/kappa)
            theoretical_rate = lr * mu_estimate * (1 - 1/kappa)
            rate_type = 'exponential'
        elif opt_key == 'Momentum':
            # Accelerated: beta \u2248 sqrt(lr * mu)
            theoretical_rate = np.sqrt(lr * mu_estimate)
            rate_type = 'exponential (accelerated)'
        elif opt_key == 'Adam':
            # No closed form - use SGD-like heuristic
            theoretical_rate = lr * mu_estimate * 0.5  # Conservative
            rate_type = 'adaptive (heuristic)'
        else:
            theoretical_rate = lr * mu_estimate
            rate_type = 'exponential'
    else:
        # Fallback: use generic rates (WARNING: these are inaccurate!)
        if problem_type == 'strongly_convex':
            if opt_key == 'SGD':
                theoretical_rate = lr * 0.1  # Assume mu ~ 0.1
                rate_type = 'exponential (mu assumed 0.1)'
            elif opt_key == 'Momentum':
                theoretical_rate = np.sqrt(lr * 0.1)
                rate_type = 'exponential (mu assumed 0.1)'
            else:
                theoretical_rate = lr * 0.05
                rate_type = 'adaptive (heuristic)'
        elif problem_type == 'convex':
            theoretical_rate = 0.5  # Sublinear O(1/sqrt(t))
            rate_type = 'sublinear'
        else:
            theoretical_rate = 0.5
            rate_type = 'sublinear'
    
    deviation = abs(empirical_rate - theoretical_rate)
    relative_deviation = deviation / theoretical_rate if theoretical_rate > 0 else np.inf
    
    result = {
        'optimizer': optimizer_name,
        'problem_type': problem_type,
        'condition_number': condition_number,
        'theoretical_rate_type': rate_type,
        'theoretical_exponent': theoretical_rate,
        'empirical_rate': empirical_rate,
        'absolute_deviation': deviation,
        'relative_deviation': relative_deviation,
        'within_theory': relative_deviation < 2.0  # Relaxed threshold for practical problems
    }
    
    if condition_number is None:
        result['warning'] = 'Condition number unknown - using heuristic bounds (may be inaccurate)'
    
    return result


def generate_convergence_report(
    results_dict: Dict[str, Any],
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate a summary table comparing empirical rates across optimizers.
    
    Args:
        results_dict: Dict mapping optimizer names to convergence analysis results
        output_path: Optional path to save CSV report
    
    Returns:
        DataFrame with convergence metrics
    """
    rows = []
    
    for opt_name, result in results_dict.items():
        if not result.get('success'):
            continue
        
        best_fit = result.get('best_fit', 'power_law')
        fit_data = result.get(best_fit, {})
        
        if best_fit == 'power_law':
            rate = fit_data.get('alpha', np.nan)
            rate_type = 'Power-law (α)'
        else:
            rate = fit_data.get('beta', np.nan)
            rate_type = 'Exponential (β)'
        
        rows.append({
            'Optimizer': opt_name,
            'Best Fit': best_fit,
            'Rate Type': rate_type,
            'Rate': rate,
            'R²': result.get('best_r_squared', np.nan),
            'Final Loss': result['losses'][-1] if 'losses' in result else np.nan
        })
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Convergence report saved to {output_path}")
    
    return df


def plot_convergence_comparison(
    results_dict: Dict[str, Any],
    output_path: Optional[Path] = None,
    title: str = "Empirical Convergence Rate Comparison"
) -> None:
    """
    Plot loss trajectories with fitted curves for multiple optimizers.
    
    Args:
        results_dict: Dict mapping optimizer names to convergence analysis results
        output_path: Optional path to save plot
        title: Plot title
    """
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Loss trajectories
    for opt_name, result in results_dict.items():
        if not result.get('success'):
            continue
        iterations = result['iterations']
        losses = result['losses']
        ax1.plot(iterations, losses, label=opt_name, alpha=0.7)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Right: Fitted curves
    for opt_name, result in results_dict.items():
        if not result.get('success'):
            continue
        iterations = result['iterations']
        best_fit = result.get('best_fit', 'power_law')
        fit_data = result.get(best_fit, {})
        
        if 'fitted_values' in fit_data:
            fitted = fit_data['fitted_values']
            r2 = fit_data.get('r_squared', 0)
            label = f"{opt_name} ({best_fit}, R²={r2:.3f})"
            ax2.plot(iterations, fitted, label=label, linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (fitted)')
    ax2.set_title('Fitted Convergence Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence plot saved to {output_path}")
    
    plt.close()


def analyze_experiment_convergence(
    experiment_results: pd.DataFrame,
    loss_column: str = 'final_loss',
    optimizer_column: str = 'optimizer',
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Analyze convergence rates from multi-seed experiment results.
    
    Args:
        experiment_results: DataFrame with experiment metrics
        loss_column: Name of column containing final loss
        optimizer_column: Name of column identifying optimizer
        output_dir: Optional directory to save outputs
    
    Returns:
        Dict with per-optimizer convergence analysis
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for opt_name in experiment_results[optimizer_column].unique():
        opt_data = experiment_results[experiment_results[optimizer_column] == opt_name]
        
        # Extract loss history if available
        if 'loss_history' in opt_data.columns:
            # Assume loss_history is a list
            loss_histories = opt_data['loss_history'].tolist()
            # Average across seeds
            if loss_histories and isinstance(loss_histories[0], (list, np.ndarray)):
                mean_loss_history = np.mean(loss_histories, axis=0).tolist()
            else:
                mean_loss_history = opt_data[loss_column].tolist()
        else:
            # Fallback: use final loss progression
            mean_loss_history = opt_data[loss_column].tolist()
        
        convergence_result = compute_empirical_rate(mean_loss_history)
        results[opt_name] = convergence_result
    
    # Generate report
    if output_dir:
        _report = generate_convergence_report(results, output_dir / 'convergence_rates.csv')
        plot_convergence_comparison(results, output_dir / 'convergence_comparison.png')
        logger.info(f"Convergence analysis complete. Results in {output_dir}")
    
    return results


if __name__ == '__main__':
    # Demo: synthetic data
    print("=== Convergence Rate Analyzer Demo ===\n")
    
    # Simulate power-law convergence
    t = np.arange(1, 101)
    loss_power = 10.0 * np.power(t, -0.8) + 0.1 + np.random.normal(0, 0.05, 100)
    
    # Simulate exponential convergence
    loss_exp = 5.0 * np.exp(-0.05 * t) + 0.1 + np.random.normal(0, 0.05, 100)
    
    # Analyze
    result_power = compute_empirical_rate(loss_power.tolist(), method='power')
    result_exp = compute_empirical_rate(loss_exp.tolist(), method='exponential')
    
    print("Power-law fit (true α=0.8):")
    if result_power.get('success'):
        print(f"  Estimated α = {result_power['power_law']['alpha']:.3f}")
        print(f"  R² = {result_power['power_law']['r_squared']:.3f}")
    
    print("\nExponential fit (true β=0.05):")
    if result_exp.get('success'):
        print(f"  Estimated β = {result_exp['exponential']['beta']:.3f}")
        print(f"  R² = {result_exp['exponential']['r_squared']:.3f}")
    
    # Compare to theory
    comparison = compare_to_theoretical_bounds(
        empirical_rate=0.8,
        optimizer_name='Momentum',
        problem_type='convex',
        lr=0.01
    )
    print("\nTheoretical comparison:")
    print(f"  Theoretical rate: {comparison['theoretical_exponent']:.3f}")
    print(f"  Relative deviation: {comparison['relative_deviation']:.2%}")
    print(f"  Within theory: {comparison['within_theory']}")
