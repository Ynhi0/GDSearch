"""
Utilities to estimate/check Polyak-Lojasiewicz (PL) condition for test functions.
"""
import numpy as np


def pl_mu_estimate(loss, grad_norm_sq, f_star=0.0, eps=1e-12):
    """Estimate local PL mu: mu_hat = ||grad||^2 / (2 * (f - f_star)).
    Returns mu_hat (np.nan if denom <= eps) and boolean whether condition holds for a provided mu_threshold (optional).
    """
    denom = 2.0 * (loss - f_star)
    if denom <= eps:
        return np.nan
    return float(grad_norm_sq / denom)


def pl_holds_at_point(loss, grad_norm_sq, mu, f_star=0.0, eps=1e-12):
    """Check whether ||grad||^2 >= 2 mu (f - f_star) (PL inequality).
    Returns bool.
    """
    lhs = grad_norm_sq
    rhs = 2.0 * mu * max(loss - f_star, eps)
    return float(lhs) >= float(rhs)


def compute_pl_over_trajectory(df, loss_col='loss', grad_norm_col='grad_norm', f_star=0.0, mu_threshold=None):
    """Given DataFrame with loss and grad_norm columns, compute mu_hat per iteration and optionally boolean column for mu_threshold.
    Returns (mu_hat_array, holds_array or None)
    """
    losses = df[loss_col].values
    grad_sq = df[grad_norm_col].values ** 2
    mu_hats = []
    holds = None
    for L, g2 in zip(losses, grad_sq):
        mu = pl_mu_estimate(L, g2, f_star=f_star)
        mu_hats.append(mu)
    mu_hats = np.array(mu_hats)
    if mu_threshold is not None:
        holds = np.array([pl_holds_at_point(L, g2, mu_threshold, f_star) for L, g2 in zip(losses, grad_sq)])
    return mu_hats, holds
