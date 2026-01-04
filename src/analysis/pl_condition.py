"""
Utilities to estimate/check Polyak-Lojasiewicz (PL) condition for test functions.

For neural networks where the global minimum is unknown, this module provides
trajectory-based estimation helpers to approximate f_star.
"""
import numpy as np
from typing import Optional, Tuple
import pandas as pd


def estimate_f_star_from_trajectory(
    losses: np.ndarray,
    method: str = 'running_min_with_margin',
    margin: float = 0.01,
    window_size: int = 100
) -> float:
    """
    Estimate f_star (global minimum) from an observed loss trajectory.

    For neural networks, the true global minimum is typically unknown and the loss
    seldom reaches zero. This helper provides several practical estimators:

    1. 'running_min': min(observed_losses)
    2. 'running_min_with_margin': min(observed_losses) scaled by (1 - margin) to avoid
       overly optimistic estimates
    3. 'windowed_min': minimum over a recent window, scaled by (1 - margin)

    Note: For neural networks, this estimates the local Polyak-Lojasiewicz constant
    relative to the best observed loss, not necessarily the global minimum.

    Args:
        losses: Array of loss values from training trajectory
        method: Estimation method ('running_min', 'running_min_with_margin', 'windowed_min')
        margin: Safety margin as fraction of min loss (default 1% = 0.01)
        window_size: Window size for 'windowed_min' method
        
    Returns:
        Estimated f_star value (always >= 0 for classification losses)
    """
    if len(losses) == 0:
        return 0.0
    
    losses_finite = losses[np.isfinite(losses)]
    if len(losses_finite) == 0:
        return 0.0
    
    min_loss = np.min(losses_finite)
    
    if method == 'running_min':
        f_star_estimate = min_loss
    elif method == 'running_min_with_margin':
        # Subtract margin to create lower bound (more conservative PL estimate)
        f_star_estimate = min_loss * (1.0 - margin)
    elif method == 'windowed_min':
        # Use only recent history
        recent_losses = losses_finite[-window_size:] if len(losses_finite) > window_size else losses_finite
        f_star_estimate = np.min(recent_losses) * (1.0 - margin)
    else:
        raise ValueError(f"Unknown f_star estimation method: {method}")
    
    # For classification, loss should be non-negative
    f_star_estimate = max(0.0, f_star_estimate)
    
    return float(f_star_estimate)


def pl_mu_estimate(
    loss: float,
    grad_norm_sq: float,
    f_star: Optional[float] = None,
    losses_trajectory: Optional[np.ndarray] = None,
    eps: float = 1e-12
) -> float:
    """
    Estimate local PL constant: mu_hat = ||grad||^2 / (2 * (f - f_star)).

    - If f_star is provided explicitly, that value is used (synthetic functions)
    - If f_star is None but losses_trajectory is provided, f_star is estimated automatically
    - Returns np.nan if the denominator is too small or inputs are invalid

    Args:
        loss: Current loss value
        grad_norm_sq: Squared gradient norm ||∇f||^2
        f_star: Known global minimum (None for neural networks)
        losses_trajectory: Full loss history for f_star estimation (required if f_star=None)
        eps: Numerical tolerance for denominator
        
    Returns:
        Estimated μ (np.nan if computation fails)
    """
    # Estimate f_star if not provided
    if f_star is None:
        if losses_trajectory is None:
            raise ValueError(
                "pl_mu_estimate: f_star is None but losses_trajectory not provided. "
                "For neural networks, you must provide the loss trajectory to estimate f_star."
            )
        f_star = estimate_f_star_from_trajectory(losses_trajectory)
    
    # Compute denominator with validation
    denom = 2.0 * (loss - f_star)
    
    if denom <= eps:
        # Loss is at or below estimated minimum (common near convergence)
        return np.nan
    
    if not np.isfinite(grad_norm_sq) or grad_norm_sq < 0:
        return np.nan
    
    mu_hat = grad_norm_sq / denom
    
    return float(mu_hat) if np.isfinite(mu_hat) else np.nan


def pl_holds_at_point(
    loss: float,
    grad_norm_sq: float,
    mu: float,
    f_star: Optional[float] = None,
    losses_trajectory: Optional[np.ndarray] = None,
    eps: float = 1e-12
) -> bool:
    """
    Check whether PL inequality holds: ||∇f||^2 >= 2μ(f - f_star).

    Args:
        loss: Current loss value
        grad_norm_sq: Squared gradient norm
        mu: PL constant to check
        f_star: Known global minimum (None for neural networks)
        losses_trajectory: Full loss history for f_star estimation
        eps: Numerical tolerance
        
    Returns:
        True if PL condition holds, False otherwise
    """
    # Estimate f_star if not provided
    if f_star is None:
        if losses_trajectory is None:
            # Legacy behavior: assume f_star=0.0 when no trajectory information is
            # available. This keeps checks permissive for synthetic tests and
            # small mathematical functions.
            f_star = 0.0
        else:
            f_star = estimate_f_star_from_trajectory(losses_trajectory)
    
    lhs = grad_norm_sq
    rhs = 2.0 * mu * max(loss - f_star, eps)
    
    return float(lhs) >= float(rhs)


def compute_pl_over_trajectory(
    df: pd.DataFrame,
    loss_col: str = 'loss',
    grad_norm_col: str = 'grad_norm',
    f_star: Optional[float] = None,
    mu_threshold: Optional[float] = None,
    auto_estimate_f_star: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute PL constant μ_hat per iteration over an entire trajectory.

    Args:
        df: DataFrame with loss and grad_norm columns
        loss_col: Name of loss column
        grad_norm_col: Name of gradient norm column
        f_star: Known global minimum (if None, will be estimated)
        mu_threshold: If provided, also return boolean array for PL condition check
        auto_estimate_f_star: If True and f_star=None, estimate from trajectory
        
    Returns:
        Tuple of:
         - mu_hat_array: Array of estimated μ values per iteration
         - holds_array: Boolean array (PL holds at each iteration) or None
    """
    losses = df[loss_col].values
    losses_array = np.asarray(losses)
    grad_norms = df[grad_norm_col].values
    grad_norms_array = np.asarray(grad_norms)
    grad_sq = grad_norms_array ** 2
    
    # Estimate f_star if needed
    if f_star is None and auto_estimate_f_star:
        f_star = estimate_f_star_from_trajectory(losses_array)
    elif f_star is None:
        # No estimation requested - use 0.0 (legacy behavior)
        f_star = 0.0
    
    # Compute mu_hat per iteration
    mu_hats = []
    for L, g2 in zip(losses_array, grad_sq):
        denom = 2.0 * (L - f_star)
        if denom > 1e-12 and np.isfinite(g2):
            mu_hat = g2 / denom
            mu_hats.append(mu_hat if np.isfinite(mu_hat) else np.nan)
        else:
            mu_hats.append(np.nan)
    
    mu_hats = np.array(mu_hats)
    
    # Check PL condition if threshold provided
    holds = None
    if mu_threshold is not None:
        holds = np.array([
            pl_holds_at_point(L, g2, mu_threshold, f_star=f_star)
            for L, g2 in zip(losses, grad_sq)
        ])
    
    return mu_hats, holds
