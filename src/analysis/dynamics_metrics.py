"""
Dynamics Metrics Module
Compute instantaneous training dynamics metrics for optimizer analysis.

This module provides functions to quantify dynamic behavior during optimization,
including trajectory smoothness, oscillation magnitude, and instantaneous speed.

Required by research proposal:
"phân tích chi tiết các đặc tính động học so sánh (độ mượt - smoothness, 
tốc độ tức thời - instantaneous rate/update magnitude, dao động - oscillations/fluctuations)"
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd


def compute_instantaneous_speed(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous speed (distance traveled per iteration).
    
    Args:
        trajectory: Array of shape (n_iterations, n_dimensions) containing parameter values
        
    Returns:
        Array of shape (n_iterations-1,) containing ||x_t - x_{t-1}|| for each step
        
    Example:
        >>> trajectory = np.array([[0, 0], [1, 0], [1, 1]])
        >>> speed = compute_instantaneous_speed(trajectory)
        >>> print(speed)  # [1.0, 1.0]
    """
    if len(trajectory) < 2:
        return np.array([])
    
    # Compute Euclidean distance between consecutive points
    diffs = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    
    return speeds


def compute_smoothness_index(trajectory: np.ndarray, window: int = 5) -> float:
    """
    Compute trajectory smoothness using angle changes between consecutive segments.
    
    Lower values indicate smoother trajectories (less zigzagging).
    Higher values indicate more oscillatory behavior.
    
    Args:
        trajectory: Array of shape (n_iterations, n_dimensions)
        window: Moving average window for smoothing
        
    Returns:
        float: Mean absolute angle change (in radians) normalized to [0, π]
        
    Theory:
        Smooth trajectories (like momentum methods) should have smaller angle changes.
        Oscillatory trajectories (like vanilla SGD) have larger angle changes.
    """
    if len(trajectory) < 3:
        return 0.0
    
    # Compute direction vectors between consecutive points
    directions = np.diff(trajectory, axis=0)
    
    # Avoid division by zero
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    directions = directions / norms
    
    # Compute angles between consecutive direction vectors
    angles = []
    for i in range(len(directions) - 1):
        # Dot product gives cos(angle)
        cos_angle = np.clip(np.dot(directions[i], directions[i+1]), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    if len(angles) == 0:
        return 0.0
    
    # Mean absolute angle change (normalized to [0, π])
    smoothness = np.mean(np.abs(angles))
    
    return smoothness


def compute_oscillation_magnitude(values: np.ndarray, ema_alpha: float = 0.1) -> np.ndarray:
    """
    Compute oscillation magnitude by measuring deviation from exponential moving average.
    
    Args:
        values: Array of metric values (e.g., loss, grad_norm) over iterations
        ema_alpha: EMA smoothing factor (0 < alpha <= 1)
                  Smaller values = more smoothing
        
    Returns:
        Array of absolute deviations from EMA (same length as input)
        
    Example:
        >>> losses = np.array([1.0, 0.5, 0.8, 0.3, 0.6, 0.2])
        >>> osc = compute_oscillation_magnitude(losses, ema_alpha=0.3)
    """
    if len(values) == 0:
        return np.array([])
    
    # Compute exponential moving average
    ema = np.zeros_like(values)
    ema[0] = values[0]
    
    for t in range(1, len(values)):
        ema[t] = ema_alpha * values[t] + (1 - ema_alpha) * ema[t-1]
    
    # Oscillation = absolute deviation from trend
    oscillations = np.abs(values - ema)
    
    return oscillations


def compute_convergence_smoothness(losses: np.ndarray, window: int = 50) -> float:
    """
    Measure convergence smoothness by fitting exponential decay and computing residuals.
    
    Args:
        losses: Training loss values over iterations
        window: Window size for moving statistics
        
    Returns:
        float: Normalized RMSE of fit (lower = smoother convergence)
    """
    if len(losses) < window:
        return float('inf')
    
    from scipy.optimize import curve_fit
    
    # Fit exponential decay: L(t) = L_inf + (L_0 - L_inf) * exp(-λt)
    def exp_decay(t, L_inf, L_0, lam):
        return L_inf + (L_0 - L_inf) * np.exp(-lam * t)
    
    try:
        iterations = np.arange(len(losses))
        # Initial guess
        p0 = [losses[-1], losses[0], 0.01]
        
        params, _ = curve_fit(exp_decay, iterations, losses, p0=p0, maxfev=5000)
        
        # Compute fit quality
        fitted = exp_decay(iterations, *params)
        residuals = losses - fitted
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # Normalize by loss range
        loss_range = losses[0] - losses[-1]
        if loss_range > 0:
            normalized_rmse = rmse / loss_range
        else:
            normalized_rmse = rmse
        
        return normalized_rmse
        
    except Exception as e:
        # Specify exception type for better error tracking
        # Fit failed - return high value
        logging.debug(f"Smoothness fit failed: {e}")
        return float('inf')


def compute_update_magnitude_stats(grad_norms: np.ndarray, 
                                   learning_rates: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics of parameter update magnitudes.
    
    Args:
        grad_norms: Gradient norms over iterations
        learning_rates: Learning rate values (may vary per iteration)
        
    Returns:
        dict: Statistics including mean, std, max update magnitude
    """
    update_magnitudes = grad_norms * learning_rates
    
    stats = {
        'mean_update': float(np.mean(update_magnitudes)),
        'std_update': float(np.std(update_magnitudes)),
        'max_update': float(np.max(update_magnitudes)),
        'min_update': float(np.min(update_magnitudes)),
        'median_update': float(np.median(update_magnitudes)),
    }
    
    return stats


def analyze_trajectory_dynamics(trajectory: np.ndarray, 
                                losses: np.ndarray,
                                grad_norms: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Comprehensive dynamics analysis of an optimization trajectory.
    
    Args:
        trajectory: Parameter values over iterations (n_iter, n_dim)
        losses: Loss values over iterations
        grad_norms: Optional gradient norms
        
    Returns:
        dict: Dictionary of dynamics metrics
    """
    metrics = {}
    
    # Trajectory metrics
    if len(trajectory) > 1:
        speeds = compute_instantaneous_speed(trajectory)
        metrics['mean_speed'] = float(np.mean(speeds))
        metrics['std_speed'] = float(np.std(speeds))
        metrics['max_speed'] = float(np.max(speeds))
        
        # Smoothness (lower = less zigzagging)
        metrics['smoothness_index'] = float(compute_smoothness_index(trajectory))
    
    # Loss dynamics
    if len(losses) > 1:
        loss_osc = compute_oscillation_magnitude(losses, ema_alpha=0.1)
        metrics['mean_loss_oscillation'] = float(np.mean(loss_osc))
        metrics['std_loss_oscillation'] = float(np.std(loss_osc))
        
        # Convergence smoothness
        metrics['convergence_smoothness'] = float(compute_convergence_smoothness(losses))
    
    # Gradient dynamics (if available)
    if grad_norms is not None and len(grad_norms) > 1:
        grad_osc = compute_oscillation_magnitude(grad_norms, ema_alpha=0.1)
        metrics['mean_grad_oscillation'] = float(np.mean(grad_osc))
        metrics['std_grad_oscillation'] = float(np.std(grad_osc))
    
    # Distance traveled
    if len(trajectory) > 1:
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        euclidean_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        metrics['total_path_length'] = float(total_distance)
        metrics['direct_distance'] = float(euclidean_distance)
        
        # Path efficiency: direct distance / total distance
        # 1.0 = straight line, < 1.0 = zigzagging
        if total_distance > 0:
            metrics['path_efficiency'] = float(euclidean_distance / total_distance)
        else:
            metrics['path_efficiency'] = 1.0
    
    return metrics


def compare_dynamics(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare dynamics metrics across multiple optimizers.
    
    Args:
        results_dict: Dictionary mapping optimizer names to their dynamics metrics
        
    Returns:
        DataFrame with comparison table
        
    Example:
        >>> results = {
        ...     'SGD': {'mean_speed': 0.1, 'smoothness_index': 0.5},
        ...     'Momentum': {'mean_speed': 0.2, 'smoothness_index': 0.3}
        ... }
        >>> df = compare_dynamics(results)
    """
    df = pd.DataFrame(results_dict).T
    
    # Sort by smoothness (lower = better)
    if 'smoothness_index' in df.columns:
        df = df.sort_values('smoothness_index')
    
    return df


if __name__ == "__main__":
    # Example usage
    logging.info("Dynamics Metrics Module - Example Usage")
    print("=" * 60)
    
    # Simulate trajectories
    np.random.seed(42)
    n_iter = 100
    
    # Smooth trajectory (momentum-like)
    smooth_traj = np.cumsum(np.random.randn(n_iter, 2) * 0.1, axis=0)
    smooth_losses = 1.0 / (np.arange(n_iter) + 1) + np.random.randn(n_iter) * 0.01
    
    # Oscillatory trajectory (SGD-like)
    osc_traj = np.cumsum(np.random.randn(n_iter, 2) * 0.5, axis=0)
    osc_losses = 1.0 / (np.arange(n_iter) + 1) + np.random.randn(n_iter) * 0.1
    
    # Analyze
    smooth_metrics = analyze_trajectory_dynamics(smooth_traj, smooth_losses)
    osc_metrics = analyze_trajectory_dynamics(osc_traj, osc_losses)
    
    # Compare
    comparison = compare_dynamics({
        'Smooth (Momentum-like)': smooth_metrics,
        'Oscillatory (SGD-like)': osc_metrics
    })
    
    logging.info("\nDynamics Comparison:")
    print(comparison[['smoothness_index', 'path_efficiency', 'mean_loss_oscillation']])
    logging.info("\n✓ Lower smoothness_index = smoother trajectory")
    logging.info("✓ Higher path_efficiency = more direct path")
    logging.info("✓ Lower loss_oscillation = more stable convergence")
