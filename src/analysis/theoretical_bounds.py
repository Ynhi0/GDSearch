"""
Theoretical analysis utilities for optimizer convergence.

Provides tools for analyzing convergence bounds, regret analysis, and complexity estimates.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging


def sgd_convergence_bound(
    L: float,
    mu: float,
    lr: float,
    T: int,
    sigma: float = 0.0
) -> Dict[str, Any]:
    """
    Compute theoretical convergence bound for SGD.
    
    For strongly convex and smooth functions:
    E[f(x_T) - f(x*)] ≤ (1 - μη)^T * ||x_0 - x*||^2 / 2 + η * σ^2 / (2μ)
    
    Args:
        L: Lipschitz smoothness constant
        mu: Strong convexity parameter
        lr: Learning rate (η)
        T: Number of iterations
        sigma: Stochastic gradient variance
        
    Returns:
        Dict containing:
         - optimal_lr: Optimal learning rate (2/(L+μ))
          - convergence_rate: Convergence rate (1 - μη)
          - iterations_to_eps: Iterations to reach ε-accuracy
          - final_bound: Expected error bound at iteration T
    """
    # Optimal learning rate for strongly convex functions
    optimal_lr = 2.0 / (L + mu) if (L + mu) > 0 else 0.0
    
    # Convergence rate (can be negative; use magnitude for stability checks)
    convergence_rate = 1.0 - mu * lr
    rate_magnitude = abs(convergence_rate)
    
    # Iterations to reach ε-accuracy (assuming deterministic case)
    # (1 - μη)^T ≤ ε  =>  T ≥ log(ε) / log(1 - μη)
    epsilon = 1e-6
    if rate_magnitude >= 1:
        # Magnitude >= 1 means no geometric decay (divergent or marginally stable)
        iterations_to_eps = float('inf')
        logging.warning(
            "sgd_convergence_bound: |convergence_rate|=%.4f >= 1. lr=%.6f, mu=%.6f."
            " Step size does not yield geometric decay.", rate_magnitude, lr, mu
        )
    else:
        # Valid convergence case: 0 < |rate| < 1 (allows bounded oscillations)
        iterations_to_eps = np.log(epsilon) / np.log(rate_magnitude)
    
    # Final error bound (simplified, assuming ||x_0 - x*||^2 = 1)
    if rate_magnitude >= 1:
        geometric_term = float('inf')  # No geometric decay
    else:
        geometric_term = rate_magnitude ** T
    
    variance_term = lr * sigma ** 2 / (2 * mu) if mu > 1e-12 else float('inf')
    
    if geometric_term == float('inf') or variance_term == float('inf'):
        final_bound = float('inf')
    else:
        final_bound = geometric_term / 2.0 + variance_term
    
    return {
        'optimal_lr': optimal_lr,
        'convergence_rate': convergence_rate,
        'iterations_to_eps': iterations_to_eps,
        'final_bound': final_bound,
        'is_lr_optimal': abs(lr - optimal_lr) / max(optimal_lr, 1e-10) < 0.1 if optimal_lr > 0 else False
    }


def momentum_convergence_bound(
    L: float,
    mu: float,
    lr: float,
    momentum: float,
    T: int,
    sigma: float = 0.0,
    method: str = 'heavy_ball'
) -> Dict[str, Any]:
    """
    Compute theoretical convergence bound for Momentum-based methods.
    
    Implements bounds for:
    1. Heavy Ball Momentum (Polyak, 1964): O((1 - sqrt(μ/L))^T) for strongly convex
    2. Nesterov Accelerated Gradient (Nesterov, 1983): O(1/T^2) for convex
    
    For strongly convex functions with Heavy Ball:
    E[f(x_T) - f(x*)] ≤ C * ρ^T, where ρ = 1 - sqrt(μ/L) (accelerated rate)
    
    For Nesterov in convex case:
    f(x_T) - f(x*) ≤ 2L||x_0 - x*||^2 / (T+1)^2
    
    Args:
        L: Lipschitz smoothness constant
        mu: Strong convexity parameter (0 for convex-only)
        lr: Learning rate (step size)
        momentum: Momentum coefficient (β)
        T: Number of iterations
        sigma: Stochastic gradient variance
        method: 'heavy_ball' (Polyak) or 'nesterov' (NAG)
        
    Returns:
        Dict containing:
          - optimal_momentum: Optimal momentum coefficient
          - convergence_rate: Convergence rate (ρ)
          - iterations_to_eps: Iterations to reach ε-accuracy
          - final_bound: Expected error bound at iteration T
          - acceleration_factor: Speedup vs vanilla SGD
          - method: Method used ('heavy_ball' or 'nesterov')
    """
    # Compute optimal momentum and learning rate for strongly convex case
    if mu > 1e-12:
        # Optimal momentum: β* = ((sqrt(κ) - 1) / (sqrt(κ) + 1))^2 where κ = L/μ
        kappa = L / mu
        sqrt_kappa = np.sqrt(kappa)
        optimal_momentum = ((sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)) ** 2
        optimal_lr = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
        
        # Accelerated convergence rate: ρ = 1 - sqrt(μ/L) for optimal parameters
        # This is faster than SGD's rate of (1 - μ/L)
        convergence_rate = 1.0 - np.sqrt(mu / L)
        
        # Vanilla SGD rate for comparison
        sgd_rate = 1.0 - mu * lr
        
        # Acceleration factor: how much faster than SGD
        if sgd_rate > 0 and sgd_rate < 1 and convergence_rate > 0:
            # Compare iteration counts: log(ε) / log(rate)
            # Lower rate = faster convergence
            acceleration_factor = np.log(sgd_rate) / np.log(convergence_rate)
        else:
            acceleration_factor = 1.0
            
    else:
        # Convex-only case (μ = 0)
        optimal_momentum = 0.0  # No strong convexity to exploit
        optimal_lr = 1.0 / L
        
        if method == 'nesterov':
            # Nesterov achieves O(1/T^2) for convex
            convergence_rate = 1.0 / (T ** 2)
        else:
            # Heavy ball without strong convexity: similar to SGD O(1/T)
            convergence_rate = 1.0 / T
        
        acceleration_factor = 1.0  # No acceleration without strong convexity
    
    # Compute iterations to reach ε-accuracy
    epsilon = 1e-6
    rate_magnitude = abs(convergence_rate)
    if method == 'nesterov' and mu <= 1e-12:
        # For Nesterov in convex case: need T such that 1/T^2 ≤ ε
        iterations_to_eps = np.sqrt(1.0 / epsilon)
    elif rate_magnitude >= 1:
        iterations_to_eps = float('inf')
        logging.warning("momentum_convergence_bound: |convergence_rate|=%.4f >= 1. "
                       "Parameters lr=%.6f, momentum=%.4f do not yield geometric decay.", 
                       rate_magnitude, lr, momentum)
    else:
        # Geometric decay case: |ρ|^T ≤ ε (allows bounded oscillations)
        iterations_to_eps = np.log(epsilon) / np.log(rate_magnitude)
    
    # Compute final error bound
    if method == 'nesterov' and mu <= 1e-12:
        # Nesterov bound: 2L||x_0 - x*||^2 / (T+1)^2
        D_squared = 1.0  # Assume ||x_0 - x*||^2 = 1
        final_bound = 2.0 * L * D_squared / ((T + 1) ** 2)
    elif mu > 1e-12:
        # Strongly convex case: C * ρ^T + variance term
        geometric_term = convergence_rate ** T if convergence_rate < 1 else float('inf')
        variance_term = lr * sigma ** 2 / (2 * mu) if mu > 1e-12 else float('inf')
        
        if geometric_term == float('inf') or variance_term == float('inf'):
            final_bound = float('inf')
        else:
            final_bound = geometric_term / 2.0 + variance_term
    else:
        # Convex case without strong convexity
        final_bound = 1.0 / T  # Simplified bound
    
    return {
        'optimal_momentum': float(optimal_momentum),
        'optimal_lr': float(optimal_lr),
        'convergence_rate': float(convergence_rate),
        'iterations_to_eps': float(iterations_to_eps),
        'final_bound': float(final_bound),
        'acceleration_factor': float(acceleration_factor),
        'is_momentum_optimal': abs(momentum - optimal_momentum) / max(optimal_momentum, 1e-10) < 0.1 if optimal_momentum > 0 else False,
        'is_lr_optimal': abs(lr - optimal_lr) / max(optimal_lr, 1e-10) < 0.1 if optimal_lr > 0 else False,
        'method': method,
        'kappa': float(L / mu) if mu > 1e-12 else float('inf')
    }


def adam_convergence_bound(
    L: float,
    T: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    alpha: float = 0.001
) -> Dict[str, Any]:
    """
    Compute theoretical regret bound for Adam.
    
    Based on "On the Convergence of Adam and Beyond" (Reddi et al., 2018).
    Regret bound: R(T) ≤ O(√T) for convex functions
    
    Args:
        L: Lipschitz constant of gradients
        T: Number of iterations
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        alpha: Step size
        epsilon: Numerical stability constant
        
    Returns:
        Dict with regret and convergence estimates
    """
    # Simplified regret bound (assumes bounded gradients)
    G = L  # Assume ||∇f|| ≤ G
    D = 1.0  # Assume ||x_0 - x*|| ≤ D
    
    # Regret bound from theory: O(√T)
    regret_bound = (alpha * G * np.sqrt(T)) / (1 - beta1)
    regret_bound += D * G / (1 - beta1) * np.sqrt(T / (1 - beta2))
    
    # Convergence rate (non-convex case)
    # E[||∇f(x_T)||^2] ≤ O(1/√T) for smooth non-convex functions
    gradient_norm_bound = L * np.sqrt(1.0 / T)
    
    return {
        'regret_bound': regret_bound,
        'per_iteration_regret': regret_bound / T,
        'gradient_norm_bound': gradient_norm_bound,
        'convergence_rate_class': 'O(1/√T)',
        'beta1': beta1,
        'beta2': beta2
    }


def sam_sharpness_bound(
    rho: float,
    L: float,
    epsilon_flat: float = 0.1
) -> Dict[str, Any]:
    """
    Analyze sharpness-aware minimization (SAM) properties.
    
    SAM seeks flat minima by minimizing:
        max_{||δ|| ≤ ρ} f(x + δ)
    
    Args:
        rho: Perturbation radius
        L: Lipschitz constant
        epsilon_flat: Target flatness (sharpness threshold)
        
    Returns:
        Dict with SAM-specific bounds
    """
    # Maximum sharpness (second-order bound)
    max_sharpness = L * rho
    
    # Effective perturbation impact
    perturbation_impact = rho * L
    
    # Number of iterations to achieve ε-flatness
    iterations_for_flatness = max_sharpness / epsilon_flat if epsilon_flat > 0 else float('inf')
    
    return {
        'perturbation_radius': rho,
        'max_sharpness': max_sharpness,
        'perturbation_impact': perturbation_impact,
        'iterations_for_flatness': iterations_for_flatness,
        'flatness_guarantee': epsilon_flat
    }


def compute_regret(
    losses: np.ndarray,
    optimal_loss: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute empirical regret from loss trajectory.
    
    Regret: R(T) = Σ_{t=1}^T [f(x_t) - f(x*)]
    
    Args:
        losses: Array of loss values over time
        optimal_loss: Known optimal loss (if available)
        
    Returns:
        Dict with regret metrics
    """
    T = len(losses)
    
    if optimal_loss is None:
        # Use minimum observed loss as proxy
        optimal_loss = np.min(losses)
    
    # Cumulative regret
    regret_per_step = losses - optimal_loss
    cumulative_regret = np.cumsum(regret_per_step)
    
    # Average regret
    avg_regret = cumulative_regret / np.arange(1, T + 1)
    
    # Regret growth rate (fit to √T or log T)
    t_vals = np.arange(1, T + 1)
    sqrt_t_fit = np.polyfit(np.sqrt(t_vals), cumulative_regret, deg=1)[0]
    log_t_fit = np.polyfit(np.log(t_vals + 1), cumulative_regret, deg=1)[0]
    
    return {
        'total_regret': cumulative_regret[-1],
        'average_regret': avg_regret[-1],
        'final_loss': losses[-1],
        'optimal_loss': optimal_loss,
        'regret_sqrt_t_coefficient': sqrt_t_fit,
        'regret_log_t_coefficient': log_t_fit,
        'sublinear_regret': cumulative_regret[-1] / T < 1.0
    }


def estimate_smoothness(
    gradients: List[np.ndarray[Any, np.dtype[np.float64]]],
    params: List[np.ndarray[Any, np.dtype[np.float64]]]
) -> float:
    """
    Estimate Lipschitz smoothness constant L from gradient samples.
    
    L ≈ max ||g_i - g_j|| / ||x_i - x_j||
    
    NUMERICAL STABILITY: Validates inputs and handles overflow/NaN gracefully.
    
    Args:
        gradients: List of gradient vectors
        params: List of parameter vectors
        
    Returns:
        Estimated L (0.0 if computation fails)
    """
    if len(gradients) < 2:
        return 0.0
    
    max_L = 0.0
    
    for i in range(len(gradients) - 1):
        for j in range(i + 1, min(i + 10, len(gradients))):  # Sample pairs
            try:
                # Validate inputs before subtraction to prevent NaN propagation
                if not (np.all(np.isfinite(gradients[i])) and np.all(np.isfinite(gradients[j]))):
                    continue  # Skip pairs with NaN/Inf
                if not (np.all(np.isfinite(params[i])) and np.all(np.isfinite(params[j]))):
                    continue
                
                grad_diff = np.linalg.norm(gradients[i] - gradients[j])
                param_diff = np.linalg.norm(params[i] - params[j])
                
                # Additional validation after norm computation
                if not np.isfinite(grad_diff) or not np.isfinite(param_diff):
                    continue
                
                if param_diff > 1e-10:
                    L_estimate = grad_diff / param_diff
                    if np.isfinite(L_estimate) and L_estimate < 1e10:  # Sanity bound
                        max_L = max(max_L, L_estimate)
            except (FloatingPointError, RuntimeWarning, ValueError):
                # Silently skip problematic pairs rather than failing entire estimation
                continue
    
    return float(max_L) if np.isfinite(max_L) else 0.0


def estimate_strong_convexity(
    gradients: List[np.ndarray[Any, np.dtype[np.float64]]],
    params: List[np.ndarray[Any, np.dtype[np.float64]]]
) -> float:
    """
    Estimate strong convexity parameter μ from gradient samples.
    
    μ ≈ min (g_i - g_j)^T (x_i - x_j) / ||x_i - x_j||^2
    
    NUMERICAL STABILITY: Validates inputs and prevents overflow in norm^2 computation.
    
    Args:
        gradients: List of gradient vectors
        params: List of parameter vectors
        
    Returns:
        Estimated μ (0.0 if computation fails)
    """
    if len(gradients) < 2:
        return 0.0
    
    min_mu = float('inf')
    
    for i in range(len(gradients) - 1):
        for j in range(i + 1, min(i + 10, len(gradients))):
            try:
                # Validate inputs before operations
                if not (np.all(np.isfinite(gradients[i])) and np.all(np.isfinite(gradients[j]))):
                    continue
                if not (np.all(np.isfinite(params[i])) and np.all(np.isfinite(params[j]))):
                    continue
                
                grad_diff = gradients[i] - gradients[j]
                param_diff = params[i] - params[j]
                
                # Validate after subtraction
                if not (np.all(np.isfinite(grad_diff)) and np.all(np.isfinite(param_diff))):
                    continue
                
                # Use safe computation: avoid x**2 overflow by computing norm then squaring with check
                param_diff_norm = np.linalg.norm(param_diff)
                if not np.isfinite(param_diff_norm) or param_diff_norm <= 1e-10:
                    continue
                    
                # Safe squared norm with overflow check
                param_diff_norm_sq = param_diff_norm * param_diff_norm
                if not np.isfinite(param_diff_norm_sq):
                    continue
                
                mu_estimate = np.dot(grad_diff, param_diff) / param_diff_norm_sq
                
                if np.isfinite(mu_estimate) and mu_estimate > 0 and mu_estimate < 1e10:
                    min_mu = min(min_mu, mu_estimate)
            except (FloatingPointError, RuntimeWarning, ValueError):
                # Silently skip problematic pairs
                continue
    
    return 0.0 if min_mu == float('inf') else float(min_mu)


def analyze_convergence_trajectory(
    losses: np.ndarray,
    optimizer_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    Comprehensive convergence analysis from loss trajectory.
    
    Args:
        losses: Array of loss values over iterations
        optimizer_name: Name of optimizer
        
    Returns:
        Dict with convergence metrics and theoretical bounds
    """
    T = len(losses)
    
    # Empirical convergence rate
    if losses[0] > losses[-1]:
        empirical_rate = (losses[0] - losses[-1]) / losses[0]
    else:
        empirical_rate = 0.0
    
    # Detect convergence regime
    recent_window = min(100, T // 4)
    recent_std = np.std(losses[-recent_window:])
    has_converged = recent_std < 0.01 * losses[-recent_window:].mean()
    
    # Compute regret
    regret_stats = compute_regret(losses)
    
    # Estimate iteration complexity (iterations to 90% of best)
    best_loss = np.min(losses)
    threshold = best_loss + 0.1 * (losses[0] - best_loss)
    iterations_to_90 = np.argmax(losses <= threshold) if np.any(losses <= threshold) else T
    
    return {
        'optimizer': optimizer_name,
        'total_iterations': int(T),
        'initial_loss': float(losses[0]),
        'final_loss': float(losses[-1]),
        'best_loss': float(best_loss),
        'empirical_rate': float(empirical_rate),
        'has_converged': bool(has_converged),
        'iterations_to_90_percent': int(iterations_to_90),
        'regret': regret_stats,
        'final_variance': float(recent_std)
    }
