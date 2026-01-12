"""
Empirical Convergence Rate Analysis Module

Computes empirical convergence rates from loss trajectories and compares them
with theoretical bounds. Supports power-law and exponential curve fitting.

This module directly supports the research proposal objective:
"Synthesis of theoretical results on convergence rate of GD/variants and
comparison with experimental observations."

CRITICAL SCIENTIFIC FIX (Gap 18):
- Model selection uses AIC (Akaike Information Criterion), NOT R²
- R² is invalid for comparing non-linear models on time-series data
- AIC penalizes model complexity: AIC = 2k - 2ln(L̂)
- Lower AIC = better model fit accounting for overfitting risk
"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_aic(residuals: np.ndarray, num_params: int) -> float:
    """
    Compute Akaike Information Criterion for model selection.

    AIC = 2k - 2ln(L̂)

    For least-squares with Gaussian errors:
    AIC = n * ln(RSS/n) + 2k

    where:
    - n = number of data points
    - RSS = residual sum of squares
    - k = number of fitted parameters

    Lower AIC = better model (balances fit quality with complexity).

    Args:
        residuals: Array of (y_true - y_pred)
        num_params: Number of parameters in the model

    Returns:
        AIC score (lower is better)

    Reference:
        Burnham & Anderson (2002), "Model Selection and Multimodel Inference"
    """
    n = len(residuals)
    rss = np.sum(residuals ** 2)

    if rss <= 0 or n <= 0:
        return np.inf

    # AIC formula for least-squares
    aic = n * np.log(rss / n) + 2 * num_params

    # Apply finite-sample correction (AICc) for small samples
    if n / num_params < 40:
        correction = (2 * num_params * (num_params + 1)) / (n - num_params - 1)
        aic += correction

    return aic


def compute_bic(residuals: np.ndarray, num_params: int) -> float:
    """
    Compute Bayesian Information Criterion for model selection.

    BIC = ln(n) * k - 2ln(L̂)

    For least-squares:
    BIC = n * ln(RSS/n) + k * ln(n)

    BIC penalizes complexity more than AIC (favors simpler models).

    Args:
        residuals: Array of (y_true - y_pred)
        num_params: Number of parameters in the model

    Returns:
        BIC score (lower is better)
    """
    n = len(residuals)
    rss = np.sum(residuals ** 2)

    if rss <= 0 or n <= 0:
        return np.inf

    bic = n * np.log(rss / n) + num_params * np.log(n)
    return bic


def fit_power_law(
    iterations: np.ndarray,
    losses: np.ndarray,
    known_min: Optional[float] = None,
    use_log_space: bool = True
) -> Dict[str, Any]:
    """
    Fit power-law convergence: loss(t) = A * t^(-α) + B

    Args:
        iterations: Array of iteration indices (1-indexed to avoid log(0))
        losses: Array of loss values
        known_min: If provided, fix B to this value (e.g., 0 for Rosenbrock)
        use_log_space: If True, fit in log-log space for better asymptotic behavior

    Returns:
        Dict with keys: alpha (convergence exponent), A, B, r_squared, success

    Scientific Note:
        - For 2D test functions with known minimum (e.g., 0), set known_min to avoid
          overfitting B and get accurate convergence rates.
        - Log-space fitting focuses on tail behavior (asymptotic regime) rather than
          early chaotic transients, which is the mathematically correct approach.
    """
    try:
        # Ensure iterations start at 1 (not 0)
        t = np.maximum(iterations, 1)

        if known_min is not None:
            # Fix B to known minimum - fit only A and alpha
            B = known_min

            if use_log_space:
                # Log-log space fitting: log(loss - B) = log(A) - alpha * log(t)
                # This is the mathematically correct way to verify power-law rates
                shifted_losses = losses - B
                # Filter out non-positive values (shouldn't happen if B is correct)
                valid_mask = shifted_losses > 0
                if np.sum(valid_mask) < 5:
                    logger.warning("Too few valid points after shifting by known_min")
                    use_log_space = False  # Fall back to linear fitting
                else:
                    log_t = np.log(t[valid_mask])
                    log_loss = np.log(shifted_losses[valid_mask])

                    # Linear regression in log-log space
                    # log_loss = log_A - alpha * log_t
                    coeffs = np.polyfit(log_t, log_loss, 1)
                    alpha = -coeffs[0]  # Slope is -alpha
                    log_A = coeffs[1]   # Intercept is log(A)
                    A = np.exp(log_A)

                    # Compute fitted values in original space
                    fitted = A * np.power(t, -alpha) + B

                    # Compute residuals and information criteria
                    residuals = losses - fitted
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((losses - np.mean(losses)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                    # GAP 18 FIX: Use AIC/BIC, not R² for model selection
                    # Power law with known B has 2 parameters: A, alpha
                    num_params = 2
                    aic = compute_aic(residuals, num_params)
                    bic = compute_bic(residuals, num_params)

                    return {
                        'alpha': alpha,
                        'A': A,
                        'B': B,
                        'r_squared': r_squared,
                        'aic': aic,
                        'bic': bic,
                        'num_params': num_params,
                        'fitted_values': fitted,
                        'success': True,
                        'fit_method': 'log-log (known B)'
                    }

            # Linear space fitting with fixed B
            def power_model_fixed(t, A, alpha):
                return A * np.power(t, -alpha) + B

            popt, pcov = optimize.curve_fit(
                power_model_fixed, t, losses,
                p0=[losses[0] - B, 0.5],
                maxfev=10000,
                bounds=([0, 0], [np.inf, 5])
            )

            A, alpha = popt
            fitted = power_model_fixed(t, A, alpha)

            # Could use pcov for uncertainty estimation in future
            _ = pcov  # Suppress unused warning

            # Compute metrics including AIC/BIC (Gap 18 fix)
            residuals = losses - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((losses - np.mean(losses)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            num_params = 2  # A, alpha
            aic = compute_aic(residuals, num_params)
            bic = compute_bic(residuals, num_params)

            return {
                'alpha': alpha,
                'A': A,
                'B': B,
                'r_squared': r_squared,
                'aic': aic,
                'bic': bic,
                'num_params': num_params,
                'fitted_values': fitted,
                'success': True,
                'fit_method': 'linear (known B)'
            }

        # Original code: fit all parameters including B
        if use_log_space:
            # Estimate B first using last few points
            B_estimate = np.mean(losses[-10:]) if len(losses) > 10 else losses[-1]
            shifted_losses = losses - B_estimate
            valid_mask = shifted_losses > 0

            if np.sum(valid_mask) >= 5:
                log_t = np.log(t[valid_mask])
                log_loss = np.log(shifted_losses[valid_mask])

                # Linear regression
                coeffs = np.polyfit(log_t, log_loss, 1)
                alpha = -coeffs[0]
                A = np.exp(coeffs[1])
                B = B_estimate

                # Refine B with nonlinear fit
                def power_model_nonlinear(t_fit, A_fit, alpha_fit, B_fit):
                    return A_fit * np.power(t_fit, -alpha_fit) + B_fit

                popt, pcov = optimize.curve_fit(
                    power_model_nonlinear, t, losses,
                    p0=[A, alpha, B],
                    maxfev=10000,
                    bounds=([0, 0, -np.inf], [np.inf, 5, np.inf])
                )

                A, alpha, B = popt
                fitted = power_model_nonlinear(t, A, alpha, B)
                _ = pcov

                residuals = losses - fitted
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((losses - np.mean(losses)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                num_params = 3  # A, alpha, B
                aic = compute_aic(residuals, num_params)
                bic = compute_bic(residuals, num_params)

                return {
                    'alpha': alpha,
                    'A': A,
                    'B': B,
                    'r_squared': r_squared,
                    'aic': aic,
                    'bic': bic,
                    'num_params': num_params,
                    'fitted_values': fitted,
                    'success': True,
                    'fit_method': 'log-log initialized'
                }

        # Fallback: original linear-space fitting
        def power_model(t_fit, A_fit, alpha_fit, B_fit):
            return A_fit * np.power(t_fit, -alpha_fit) + B_fit

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

        # Compute metrics including AIC/BIC (Gap 18 fix)
        residuals = losses - fitted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        num_params = 3  # A, alpha, B
        aic = compute_aic(residuals, num_params)
        bic = compute_bic(residuals, num_params)

        return {
            'alpha': alpha,
            'A': A,
            'B': B,
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'num_params': num_params,
            'fitted_values': fitted,
            'success': True,
            'fit_method': 'linear (all params)'
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("Power-law fit failed: %s", e)
        return {'success': False, 'error': str(e)}


def fit_exponential(
    iterations: np.ndarray,
    losses: np.ndarray,
    known_min: Optional[float] = None,
    use_log_space: bool = True
) -> Dict[str, Any]:
    """
    Fit exponential convergence: loss(t) = A * exp(-β * t) + B

    Args:
        iterations: Array of iteration indices
        losses: Array of loss values
        known_min: If provided, fix B to this value
        use_log_space: If True, fit in log-linear space for better asymptotic behavior

    Returns:
        Dict with keys: beta (convergence rate), A, B, r_squared, success

    Scientific Note:
        - Log-linear fitting: log(loss - B) = log(A) - beta * t
        - This focuses on tail behavior (asymptotic regime)
    """
    try:
        t = iterations

        if known_min is not None:
            # Fix B to known minimum
            B = known_min

            if use_log_space:
                shifted_losses = losses - B
                valid_mask = shifted_losses > 0

                if np.sum(valid_mask) < 5:
                    logger.warning("Too few valid points after shifting by known_min")
                    use_log_space = False
                else:
                    log_loss = np.log(shifted_losses[valid_mask])
                    t_valid = t[valid_mask]

                    # Linear regression: log_loss = log_A - beta * t
                    coeffs = np.polyfit(t_valid, log_loss, 1)
                    beta = -coeffs[0]  # Slope is -beta
                    log_A = coeffs[1]   # Intercept is log(A)
                    A = np.exp(log_A)

                    fitted = A * np.exp(-beta * t) + B

                    residuals = losses - fitted
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((losses - np.mean(losses)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                    num_params = 2  # A, beta
                    aic = compute_aic(residuals, num_params)
                    bic = compute_bic(residuals, num_params)

                    return {
                        'beta': beta,
                        'A': A,
                        'B': B,
                        'r_squared': r_squared,
                        'aic': aic,
                        'bic': bic,
                        'num_params': num_params,
                        'fitted_values': fitted,
                        'success': True,
                        'fit_method': 'log-linear (known B)'
                    }

            # Linear space fitting with fixed B
            def exp_model_fixed(t, A, beta):
                return A * np.exp(-beta * t) + B

            popt, pcov = optimize.curve_fit(
                exp_model_fixed, t, losses,
                p0=[losses[0] - B, 0.01],
                maxfev=10000,
                bounds=([0, 0], [np.inf, 1])
            )

            A, beta = popt
            fitted = exp_model_fixed(t, A, beta)
            _ = pcov

            residuals = losses - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((losses - np.mean(losses)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            num_params = 2  # A, beta
            aic = compute_aic(residuals, num_params)
            bic = compute_bic(residuals, num_params)

            return {
                'beta': beta,
                'A': A,
                'B': B,
                'r_squared': r_squared,
                'aic': aic,
                'bic': bic,
                'num_params': num_params,
                'fitted_values': fitted,
                'success': True,
                'fit_method': 'linear (known B)'
            }

        # Original code: fit all parameters
        if use_log_space:
            B_estimate = np.mean(losses[-10:]) if len(losses) > 10 else losses[-1]
            shifted_losses = losses - B_estimate
            valid_mask = shifted_losses > 0

            if np.sum(valid_mask) >= 5:
                log_loss = np.log(shifted_losses[valid_mask])
                t_valid = t[valid_mask]

                coeffs = np.polyfit(t_valid, log_loss, 1)
                beta = -coeffs[0]
                A = np.exp(coeffs[1])
                B = B_estimate

                # Refine with nonlinear fit
                def exp_model_nonlinear(t_fit, A_fit, beta_fit, B_fit):
                    return A_fit * np.exp(-beta_fit * t_fit) + B_fit

                popt, pcov = optimize.curve_fit(
                    exp_model_nonlinear, t, losses,
                    p0=[A, beta, B],
                    maxfev=10000,
                    bounds=([0, 0, -np.inf], [np.inf, 1, np.inf])
                )

                A, beta, B = popt
                fitted = exp_model_nonlinear(t, A, beta, B)
                _ = pcov

                residuals = losses - fitted
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((losses - np.mean(losses)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                num_params = 3  # A, beta, B
                aic = compute_aic(residuals, num_params)
                bic = compute_bic(residuals, num_params)

                return {
                    'beta': beta,
                    'A': A,
                    'B': B,
                    'r_squared': r_squared,
                    'aic': aic,
                    'bic': bic,
                    'num_params': num_params,
                    'fitted_values': fitted,
                    'success': True,
                    'fit_method': 'log-linear initialized'
                }

        # Fallback: original linear-space fitting
        def exp_model(t_fit, A_fit, beta_fit, B_fit):
            return A_fit * np.exp(-beta_fit * t_fit) + B_fit

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

        # Compute metrics including AIC/BIC (Gap 18 fix)
        residuals = losses - fitted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        num_params = 3  # A, beta, B
        aic = compute_aic(residuals, num_params)
        bic = compute_bic(residuals, num_params)

        return {
            'beta': beta,
            'A': A,
            'B': B,
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'num_params': num_params,
            'fitted_values': fitted,
            'success': True,
            'fit_method': 'linear (all params)'
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("Exponential fit failed: %s", e)
        return {'success': False, 'error': str(e)}


def compute_empirical_rate(
    loss_history: List[float],
    method: str = 'auto',
    known_min: Optional[float] = None,
    use_log_space: bool = True
) -> Dict[str, Any]:
    """
    Compute empirical convergence rate from loss trajectory.

    Args:
        loss_history: List of loss values over training
        method: 'power', 'exponential', or 'auto' (tries both)
        known_min: For 2D functions with known minimum (e.g., 0 for Rosenbrock),
                   fix B to this value for more accurate rate estimation
        use_log_space: Use log-space fitting for better asymptotic behavior

    Returns:
        Dict with convergence metrics and best-fit model

    Example:
        # For 2D Rosenbrock (known minimum = 0)
        rate = compute_empirical_rate(losses, known_min=0.0)

        # For neural networks (unknown minimum)
        rate = compute_empirical_rate(losses)
    """
    if len(loss_history) < 10:
        return {'success': False, 'error': 'Insufficient data (<10 points)'}

    iterations = np.arange(len(loss_history))
    losses = np.array(loss_history)

    results = {'iterations': iterations, 'losses': losses}

    # Initialize to None to avoid unbound variable errors
    power_fit = None
    exp_fit = None

    if method in ['power', 'auto']:
        power_fit = fit_power_law(
            iterations + 1, losses,
            known_min=known_min,
            use_log_space=use_log_space
        )
        results['power_law'] = power_fit

    if method in ['exponential', 'auto']:
        exp_fit = fit_exponential(
            iterations, losses,
            known_min=known_min,
            use_log_space=use_log_space
        )
        results['exponential'] = exp_fit

    # GAP 18 FIX: Select best fit based on AIC (not R²)
    # AIC accounts for model complexity and prevents overfitting
    # Lower AIC = better model
    if method == 'auto':
        power_aic = power_fit.get('aic', np.inf) if power_fit and power_fit.get('success') else np.inf
        exp_aic = exp_fit.get('aic', np.inf) if exp_fit and exp_fit.get('success') else np.inf

        # CRITICAL: Select model with LOWEST AIC (not highest R²)
        if power_aic < exp_aic:
            results['best_fit'] = 'power_law'
            results['best_aic'] = power_aic
            results['selection_criterion'] = 'AIC (Akaike Information Criterion)'
            results['aic_delta'] = exp_aic - power_aic  # How much better power_law is
        else:
            results['best_fit'] = 'exponential'
            results['best_aic'] = exp_aic
            results['selection_criterion'] = 'AIC (Akaike Information Criterion)'
            results['aic_delta'] = power_aic - exp_aic  # How much better exponential is

        # Keep R² for reference (but DON'T use for model selection)
        power_r2 = power_fit.get('r_squared', -1) if power_fit and power_fit.get('success') else -1
        exp_r2 = exp_fit.get('r_squared', -1) if exp_fit and exp_fit.get('success') else -1
        results['power_r2'] = power_r2
        results['exp_r2'] = exp_r2

    elif method == 'power':
        results['best_fit'] = 'power_law'
        results['best_aic'] = power_fit.get('aic', np.inf) if power_fit else np.inf
        results['best_r_squared'] = power_fit.get('r_squared', 0) if power_fit else 0
    else:
        results['best_fit'] = 'exponential'
        results['best_aic'] = exp_fit.get('aic', np.inf) if exp_fit else np.inf
        results['best_r_squared'] = exp_fit.get('r_squared', 0) if exp_fit else 0

    results['success'] = True
    return results


def compare_to_theoretical_bounds(
    empirical_rate: float,
    optimizer_name: str,
    problem_type: str = 'strongly_convex',
    lr: float = 0.01,
    condition_number: Optional[float] = None,
    T: int = 1000
) -> Dict[str, Any]:
    """
    Compare empirical convergence rate to theoretical bounds.

    Properly accounts for problem condition number (kappa = L/mu)
    in theoretical rate predictions. Without this, theoretical bounds are
    meaningless as they ignore the problem geometry.

    Correct theoretical rates:
    - SGD (strongly convex): beta = lr * mu * (1 - 1/kappa) where mu = strong convexity
    - Momentum: beta ≈ sqrt(lr * mu)
    - Adam: No closed-form bound (adaptive)
    - Non-convex: Gradient norm bounds E[||∇f||²] ≤ O(1/√T) with proper constants

    Args:
        empirical_rate: Measured convergence exponent (α for power-law or β for exponential)
        optimizer_name: Name of optimizer (SGD, Adam, etc.)
        problem_type: 'strongly_convex', 'convex', or 'non_convex'
        lr: Learning rate used
        condition_number: kappa = L/mu (eigenvalue ratio for quadratics)
        T: Number of iterations (required for non-convex bounds)

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

    # Compute theoretical rate using condition number
    if condition_number is not None and problem_type == 'strongly_convex':
        kappa = condition_number
        # Estimate mu: for normalized problems, assume L ~ 1, so mu ~ 1/kappa
        mu_estimate = 1.0 / kappa

        # GAP FIX #9: Clarify that linear convergence rate applies to DETERMINISTIC GD
        # or to the TRANSIENT PHASE of SGD (before hitting noise floor)
        # True constant-step-size SGD converges to a noise ball, not exponentially to optimum
        if opt_key == 'SGD':
            # Linear convergence: (1 - lr*mu) per iteration
            # CRITICAL NOTE: This is the TRANSIENT PHASE rate (before noise floor)
            # Full SGD with constant step size: f(x_t) → f* + O(ησ²/(2μ))
            # The O(ησ²/(2μ)) term is the noise floor (stationary phase)
            theoretical_rate = lr * mu_estimate * (1 - 1/kappa)
            rate_type = 'exponential (transient phase only - see noise floor caveat)'
        elif opt_key == 'Momentum':
            # Accelerated: beta \u2248 sqrt(lr * mu)
            theoretical_rate = np.sqrt(lr * mu_estimate)
            rate_type = 'exponential (accelerated)'
        elif opt_key == 'Adam':
            # Use non-convex gradient norm bound (import from theoretical_bounds)
            # E[||∇f||²] ≤ 2Δ/(ηT) + Lησ²
            # For convergence rate comparison, use 1/√T form
            theoretical_rate = 0.5 * np.sqrt(lr)  # O(1/√T) with lr scaling
            rate_type = 'gradient_norm (non-convex)'
        else:
            theoretical_rate = lr * mu_estimate
            rate_type = 'exponential'
    else:
        # Fallback: use generic rates based on problem type
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
            # Convex (non-strongly): O(1/T) for SGD, O(1/T²) for Nesterov
            if opt_key in ['Momentum', 'Nesterov']:
                theoretical_rate = 2.0 / T if T > 0 else 1.0  # O(1/T²) ≈ 2/T for power law
            else:
                theoretical_rate = 1.0 / T if T > 0 else 1.0  # O(1/T)
            rate_type = 'sublinear (1/T or 1/T²)'
        else:
            # Non-convex: Use actual theoretical bounds with proper constants
            # Import bound functions for correct implementation
            from src.analysis.theoretical_bounds import sgd_convergence_bound, momentum_convergence_bound, adam_convergence_bound

            # Estimate L if not available (common for neural networks: L ~ 10-100)
            L_estimate = 10.0

            if opt_key == 'SGD':
                sgd_result = sgd_convergence_bound(
                    L=L_estimate,
                    mu=0.0,  # Non-convex
                    lr=lr,
                    T=max(T, 100),
                    sigma=0.01,  # Assume small stochastic variance
                    problem_type='non_convex'
                )
                # Gradient norm bound: E[||∇f||²] ≤ 2Δ/(ηT) + Lησ²
                # Extract coefficient for 1/√T scaling
                theoretical_rate = np.sqrt(sgd_result['final_bound'])  # √(||∇f||²)
                rate_type = 'gradient_norm (non-convex)'
            elif opt_key == 'Momentum':
                mom_result = momentum_convergence_bound(
                    L=L_estimate,
                    mu=0.0,  # Non-convex
                    lr=lr,
                    momentum=0.9,  # Typical value
                    T=max(T, 100),
                    method='heavy_ball'
                )
                # Momentum achieves O(1/T) in non-convex (Jin et al. 2017)
                theoretical_rate = mom_result['convergence_rate']  # 1/T coefficient
                rate_type = 'gradient_norm (non-convex, momentum-accelerated)'
            elif opt_key == 'Adam':
                adam_result = adam_convergence_bound(
                    L=L_estimate,
                    T=max(T, 100),
                    alpha=lr,
                    problem_type='non_convex'
                )
                # Adam gradient norm bound: E[||∇f||²] ≤ C₁/√T + C₂
                theoretical_rate = adam_result['gradient_norm_bound']
                rate_type = 'gradient_norm (non-convex, adaptive)'
            else:
                # Generic adaptive optimizer
                adam_result = adam_convergence_bound(
                    L=L_estimate,
                    T=max(T, 100),
                    alpha=lr,
                    problem_type='non_convex'
                )
                theoretical_rate = adam_result['gradient_norm_bound']
                rate_type = 'gradient_norm (non-convex, adaptive fallback)'


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
        logger.info("Convergence report saved to %s", output_path)

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
        logger.info("Convergence plot saved to %s", output_path)

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
        logger.info("Convergence analysis complete. Results in %s", output_dir)

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
