"""
Condition-Number Controlled Test Functions for Geometric Analysis.

Addresses the methodological flaw of testing on fixed functions
(Rosenbrock, Ackley) without systematic control of geometric properties.

This module provides test functions with TUNABLE condition numbers, allowing
rigorous analysis of "Reaction to different geometric structures" and
systematic validation that Momentum handles ill-conditioned problems better
than vanilla SGD.

Key Insight: Momentum's acceleration benefit is proportional to sqrt(κ),
where κ = L/μ is the condition number. To prove this, we must sweep κ.
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional
import logging


def quadratic_with_condition_number(
    kappa: float,
    n_dims: int = 2,
    random_rotation: bool = False,
    seed: Optional[int] = None
) -> Tuple[Callable, Callable, Dict[str, Any]]:
    """
    Create a quadratic function with specified condition number.

    The function is:
        f(x) = 0.5 * x^T Q x
    where Q is a positive definite matrix with eigenvalues [1, κ].

    This allows SYSTEMATIC analysis of how optimizers perform as a function
    of problem conditioning (κ), which is REQUIRED to validate theoretical
    claims about Momentum's advantage.

    Args:
        kappa: Condition number (λ_max / λ_min). Must be >= 1.
        n_dims: Number of dimensions
        random_rotation: If True, randomly rotate coordinate system
        seed: Random seed for rotation

    Returns:
        Tuple of:
         - f: Function that takes x (numpy array) and returns scalar loss
          - grad_f: Function that takes x and returns gradient
          - metadata: Dict with theoretical properties (L, μ, κ, optimal_x)
    """
    if kappa < 1.0:
        raise ValueError(f"Condition number must be >= 1, got {kappa}")

    # Create eigenvalues: linear spacing from 1 to κ
    eigenvalues = np.linspace(1.0, kappa, n_dims)

    # Create diagonal matrix Q
    Q = np.diag(eigenvalues)

    # Apply random rotation if requested
    if random_rotation:
        if seed is not None:
            np.random.seed(seed)
        U, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
        Q = U.T @ Q @ U  # Rotate: Q' = U^T Q U

    # Theoretical properties
    L = kappa  # λ_max (Lipschitz constant)
    mu = 1.0   # λ_min (strong convexity)
    optimal_x = np.zeros(n_dims)
    optimal_value = 0.0

    def f(x: np.ndarray) -> float:
        """Evaluate f(x) = 0.5 * x^T Q x"""
        return float(0.5 * x @ Q @ x)

    def grad_f(x: np.ndarray) -> np.ndarray:
        """Evaluate gradient: ∇f(x) = Q x"""
        return Q @ x

    metadata = {
        'L': float(L),
        'mu': float(mu),
        'kappa': float(kappa),
        'optimal_x': optimal_x,
        'optimal_value': optimal_value,
        'function_type': 'quadratic',
        'Q_matrix': Q,
        'eigenvalues': eigenvalues,
        'is_rotated': random_rotation
    }

    return f, grad_f, metadata


def logistic_regression_with_condition_number(
    kappa: float,
    n_samples: int = 100,
    n_features: int = 10,
    seed: Optional[int] = 42
) -> Tuple[Callable, Callable, Dict[str, Any]]:
    """
    Create a logistic regression problem with controlled condition number.

    Generates synthetic classification data where the feature covariance matrix
    has condition number κ. This provides a more realistic (non-convex) test
    compared to pure quadratics.

    Args:
        kappa: Desired condition number of feature covariance
        n_samples: Number of training samples
        n_features: Number of features
        seed: Random seed

    Returns:
        Tuple of (loss_fn, grad_fn, metadata)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate covariance matrix with controlled condition number
    eigenvalues = np.linspace(1.0, kappa, n_features)
    U, _ = np.linalg.qr(np.random.randn(n_features, n_features))
    Sigma = U.T @ np.diag(eigenvalues) @ U

    # Generate features from multivariate normal
    X = np.random.multivariate_normal(np.zeros(n_features), Sigma, size=n_samples)

    # Generate true weights and labels
    w_true = np.random.randn(n_features)
    w_true /= np.linalg.norm(w_true)

    logits = X @ w_true
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = (probabilities > 0.5).astype(float)

    def loss_fn(w: np.ndarray) -> float:
        """Binary cross-entropy loss"""
        logits = X @ w
        # Numerically stable sigmoid
        pos_mask = logits >= 0
        neg_mask = ~pos_mask

        loss = np.zeros_like(logits)
        loss[pos_mask] = np.log(1 + np.exp(-logits[pos_mask]))
        loss[neg_mask] = -logits[neg_mask] + np.log(1 + np.exp(logits[neg_mask]))

        return float(np.mean(y * loss + (1 - y) * loss))

    def grad_fn(w: np.ndarray) -> np.ndarray:
        """Gradient of loss"""
        logits = X @ w
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        errors = probabilities - y
        return X.T @ errors / n_samples

    metadata = {
        'kappa': float(kappa),
        'n_samples': n_samples,
        'n_features': n_features,
        'function_type': 'logistic_regression',
        'X': X,
        'y': y,
        'w_true': w_true,
        'covariance_eigenvalues': eigenvalues
    }

    return loss_fn, grad_fn, metadata


def sweep_condition_number_experiment(
    kappa_values: list,
    optimizer_configs: Dict[str, Dict],
    n_iterations: int = 1000,
    n_dims: int = 10,
    n_seeds: int = 3,
    initial_distance: float = 1.0
) -> Dict[str, Any]:
    """
    Run systematic experiment sweeping condition number.

    This is the KEY EXPERIMENT to validate Momentum's theoretical advantage:
    Plot "Iterations to Convergence" vs "Condition Number" for SGD and Momentum.
    Theory predicts Momentum scales as O(sqrt(κ)) while SGD scales as O(κ).

    Args:
        kappa_values: List of condition numbers to test
        optimizer_configs: Dict mapping optimizer names to config dicts
        n_iterations: Maximum iterations per run
        n_dims: Problem dimensionality
        n_seeds: Number of random seeds per configuration
        initial_distance: Initial distance from optimum

    Returns:
        Dict containing results DataFrame and convergence plots
    """
    results = []

    for kappa in kappa_values:
        logging.info(f"Testing condition number κ = {kappa:.1f}")

        for seed in range(n_seeds):
            # Create test function
            f, grad_f, metadata = quadratic_with_condition_number(
                kappa, n_dims=n_dims, random_rotation=True, seed=seed
            )

            for opt_name, opt_config in optimizer_configs.items():
                # Initialize at fixed distance from optimum
                x0 = np.random.randn(n_dims)
                x0 = x0 / np.linalg.norm(x0) * initial_distance

                # Run optimizer
                trajectory = run_optimizer_on_function(
                    f, grad_f, x0, opt_config, n_iterations, metadata
                )

                # Compute convergence metrics
                final_loss = trajectory['losses'][-1]
                iterations_to_eps = compute_iterations_to_convergence(
                    trajectory['losses'], epsilon=1e-6, optimal_value=0.0
                )

                results.append({
                    'kappa': kappa,
                    'optimizer': opt_name,
                    'seed': seed,
                    'iterations_to_convergence': iterations_to_eps,
                    'final_loss': final_loss,
                    'theoretical_rate_sgd': kappa,  # O(κ)
                    'theoretical_rate_momentum': np.sqrt(kappa),  # O(sqrt(κ))
                    **trajectory['summary']
                })

    import pandas as pd
    df = pd.DataFrame(results)

    return {
        'results_df': df,
        'kappa_values': kappa_values,
        'optimizer_configs': optimizer_configs
    }


def run_optimizer_on_function(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    optimizer_config: Dict,
    n_iterations: int,
    metadata: Dict
) -> Dict[str, Any]:
    """
    Run a simple optimizer on a test function.

    Args:
        f: Loss function
        grad_f: Gradient function
        x0: Initial point
        optimizer_config: Dict with 'type', 'lr', 'momentum', etc.
        n_iterations: Maximum iterations
        metadata: Function metadata (for L, μ, etc.)

    Returns:
        Dict with trajectory data
    """
    x = x0.copy()
    losses = []
    grad_norms = []
    distances = []

    opt_type = optimizer_config.get('type', 'sgd')
    lr = optimizer_config.get('lr', 0.01)
    momentum = optimizer_config.get('momentum', 0.0)

    # Momentum state
    velocity = np.zeros_like(x)

    optimal_x = metadata.get('optimal_x', np.zeros_like(x))

    for _ in range(n_iterations):
        # Compute loss and gradient
        loss = f(x)
        grad = grad_f(x)

        # Track metrics
        losses.append(loss)
        grad_norms.append(np.linalg.norm(grad))
        distances.append(np.linalg.norm(x - optimal_x))

        # Update step
        if opt_type == 'sgd' and momentum > 0:
            # SGD with momentum
            velocity = momentum * velocity - lr * grad
            x = x + velocity
        elif opt_type == 'sgd':
            # Vanilla SGD
            x = x - lr * grad
        elif opt_type == 'adam':
            # Simplified Adam (not fully implemented here)
            x = x - lr * grad
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        # Early stopping
        if loss < 1e-12:
            break

    return {
        'losses': np.array(losses),
        'grad_norms': np.array(grad_norms),
        'distances': np.array(distances),
        'final_x': x,
        'iterations': len(losses),
        'summary': {
            'converged': losses[-1] < 1e-6,
            'final_grad_norm': grad_norms[-1]
        }
    }


def compute_iterations_to_convergence(
    losses: np.ndarray,
    epsilon: float = 1e-6,
    optimal_value: float = 0.0
) -> int:
    """
    Compute number of iterations to reach ε-accuracy.

    Returns iteration index where |f(x_t) - f*| <= ε for the first time.
    """
    errors = np.abs(losses - optimal_value)
    converged_mask = errors <= epsilon

    if not np.any(converged_mask):
        return len(losses)  # Did not converge

    return int(np.argmax(converged_mask))


def visualize_condition_number_sweep(
    results_df,
    output_path: str = 'condition_number_sweep.png'
):
    """
    Create publication-quality plot of convergence vs condition number.

    This plot is CRITICAL for validating Momentum's theoretical advantage.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for optimizer in results_df['optimizer'].unique():
        df_opt = results_df[results_df['optimizer'] == optimizer]

        # Aggregate over seeds
        aggregated = df_opt.groupby('kappa')['iterations_to_convergence'].agg(['mean', 'std'])

        plt.plot(aggregated.index, aggregated['mean'], marker='o', label=optimizer, linewidth=2)
        plt.fill_between(
            aggregated.index,
            aggregated['mean'] - aggregated['std'],
            aggregated['mean'] + aggregated['std'],
            alpha=0.2
        )

    # Add theoretical scaling lines
    kappa_range = np.array(sorted(results_df['kappa'].unique()))
    plt.plot(kappa_range, kappa_range / kappa_range[0] * 10, '--',
             label='O(κ) - SGD theory', color='gray', alpha=0.5)
    plt.plot(kappa_range, np.sqrt(kappa_range) / np.sqrt(kappa_range[0]) * 10, '--',
             label='O(√κ) - Momentum theory', color='black', alpha=0.5)

    plt.xlabel('Condition Number (κ)', fontsize=14)
    plt.ylabel('Iterations to Convergence', fontsize=14)
    plt.title('Optimizer Scaling with Problem Conditioning', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved condition number sweep plot to {output_path}")
