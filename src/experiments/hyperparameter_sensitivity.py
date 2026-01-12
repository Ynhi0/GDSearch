"""
Hyperparameter Sensitivity Analysis for Optimizer Dynamics

Systematically analyzes the effect of key hyperparameters (β, β1, β2) on:
- Convergence trajectory
- Update magnitude
- Oscillation patterns
- Convergence rate

This addresses the research proposal's requirement for detailed dynamics analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    from src.core.test_functions import Rosenbrock, Ackley2D
    from src.core.optimizers import SGD, SGDMomentum, Adam, SGDNesterov
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.test_functions import Rosenbrock, Ackley2D
    from src.core.optimizers import SGD, SGDMomentum, Adam, SGDNesterov


def compute_trajectory_smoothness(trajectory: np.ndarray) -> float:
    """
    Compute trajectory smoothness as inverse of curvature.

    Returns:
        smoothness: Mean cosine similarity between consecutive updates
    """
    if len(trajectory) < 3:
        return 1.0

    # Compute consecutive update vectors
    updates = np.diff(trajectory, axis=0)

    # Compute angles between consecutive updates
    angles = []
    for i in range(len(updates) - 1):
        u1, u2 = updates[i], updates[i+1]
        norm1, norm2 = np.linalg.norm(u1), np.linalg.norm(u2)
        if norm1 > 1e-10 and norm2 > 1e-10:
            cos_angle = np.dot(u1, u2) / (norm1 * norm2)
            angles.append(np.clip(cos_angle, -1.0, 1.0))

    # Smoothness = high cosine similarity (aligned updates)
    smoothness = float(np.mean(angles)) if angles else 1.0
    return smoothness


def compute_oscillation_index(trajectory: np.ndarray) -> float:
    """
    Compute oscillation index based on direction changes.

    Returns:
        oscillation_index: Fraction of steps with direction reversals
    """
    if len(trajectory) < 3:
        return 0.0

    updates = np.diff(trajectory, axis=0)

    # Count direction reversals
    reversals = 0
    for i in range(len(updates) - 1):
        u1, u2 = updates[i], updates[i+1]
        norm1, norm2 = np.linalg.norm(u1), np.linalg.norm(u2)
        if norm1 > 1e-10 and norm2 > 1e-10:
            cos_angle = np.dot(u1, u2) / (norm1 * norm2)
            if cos_angle < 0:  # Direction reversal
                reversals += 1

    oscillation_index = float(reversals) / max(1, len(updates) - 1)
    return float(oscillation_index)


def run_optimizer_trajectory(optimizer, func, grad_func, x0, max_iters=1000, tol=1e-6):
    """
    Run optimizer and collect trajectory data.

    Returns:
        trajectory: List of (x, y) positions
        losses: List of function values
        grad_norms: List of gradient norms
    """
    trajectory = [x0]
    losses = [func(*x0)]
    grad_norms = []

    params = np.array(x0, dtype=float)

    for _ in range(max_iters):
        grad = grad_func(*params)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)

        if grad_norm < tol:
            break

        params = optimizer.step(params, grad)
        trajectory.append(params.copy())
        losses.append(func(*params))

    return np.array(trajectory), np.array(losses), np.array(grad_norms)


def momentum_beta_sweep(
    test_function='rosenbrock',
    beta_values=np.linspace(0.0, 0.99, 11),
    lr=0.01,
    x0=np.array([-1.5, 2.0]),
    max_iters=1000,
    output_dir='results/hyperparameter_sensitivity',
    use_coupled_lr=False,
    noise_std: float = 0.1  # GAP FIX: Add noise for realistic momentum analysis
):
    """
    Systematic sweep of momentum β parameter.

    Addresses research proposal requirement: "khảo sát hệ thống ảnh hưởng của
    các siêu tham số đặc trưng (β cho Momentum)"

    GAP FIX: Added noise_std parameter (default 0.1).
    The primary role of Momentum (β) is to dampen STOCHASTIC NOISE.
    Testing on deterministic functions (noise=0) is scientifically invalid:
    - High β might cause oscillation on smooth surfaces
    - But high β REDUCES oscillation in noisy settings (by averaging)
    - Conclusions without noise will be OPPOSITE to reality for SGD!

    SCIENTIFIC NOTE - Beta vs. Learning Rate Coupling:
    The effective step size in momentum is approximately lr/(1-β).
    When use_coupled_lr=False (default), uses FIXED lr for all β values.
    This confounds magnitude effects with dynamical effects.

    When use_coupled_lr=True, scales lr by (1-β) to maintain constant effective
    step size, isolating the momentum dynamics from magnitude scaling.

    Args:
        use_coupled_lr: If True, scale lr by (1-beta) to decouple magnitude from dynamics
        noise_std: Gradient noise std dev (default 0.1 for realistic SGD analysis)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Select test function
    if test_function == 'rosenbrock':
        test_fn = Rosenbrock()
    else:
        test_fn = Ackley2D()

    func = test_fn.compute
    # GAP FIX: Pass noise_std to gradient for realistic sensitivity analysis
    grad_func = lambda x, y: np.array(test_fn.gradient(x, y, noise_std=noise_std))

    results = []

    for beta in beta_values:
        # Apply coupled learning rate to isolate momentum dynamics
        effective_lr = lr * (1.0 - beta) if use_coupled_lr else lr
        optimizer = SGDMomentum(lr=effective_lr, beta=beta)

        trajectory, losses, grad_norms = run_optimizer_trajectory(
            optimizer, func, grad_func, x0, max_iters
        )

        # Compute dynamics metrics
        smoothness = compute_trajectory_smoothness(trajectory)
        oscillation = compute_oscillation_index(trajectory)
        final_loss = losses[-1]
        convergence_iters = len(losses)

        # Compute mean update magnitude
        if len(trajectory) > 1:
            updates = np.diff(trajectory, axis=0)
            mean_update_mag = np.mean([np.linalg.norm(u) for u in updates])
        else:
            mean_update_mag = 0.0

        results.append({
            'beta': beta,
            'effective_lr': effective_lr,
            'final_loss': final_loss,
            'convergence_iters': convergence_iters,
            'smoothness': smoothness,
            'oscillation_index': oscillation,
            'mean_update_magnitude': mean_update_mag,
            'final_grad_norm': grad_norms[-1] if len(grad_norms) > 0 else np.nan
        })

        logging.info(f"β={beta:.2f}, effective_lr={effective_lr:.6f}: loss={final_loss:.6f}, iters={convergence_iters}, "
                    f"smoothness={smoothness:.3f}, oscillation={oscillation:.3f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / f'momentum_beta_sweep_{test_function}.csv', index=False)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(df['beta'], df['final_loss'], 'o-')
    axes[0, 0].set_xlabel('Momentum β')
    axes[0, 0].set_ylabel('Final Loss')
    axes[0, 0].set_title('Convergence Quality vs β')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['beta'], df['convergence_iters'], 'o-', color='orange')
    axes[0, 1].set_xlabel('Momentum β')
    axes[0, 1].set_ylabel('Iterations to Converge')
    axes[0, 1].set_title('Convergence Speed vs β')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(df['beta'], df['smoothness'], 'o-', color='green')
    axes[0, 2].set_xlabel('Momentum β')
    axes[0, 2].set_ylabel('Trajectory Smoothness')
    axes[0, 2].set_title('Dynamics: Smoothness vs β')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(df['beta'], df['oscillation_index'], 'o-', color='red')
    axes[1, 0].set_xlabel('Momentum β')
    axes[1, 0].set_ylabel('Oscillation Index')
    axes[1, 0].set_title('Dynamics: Oscillation vs β')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df['beta'], df['mean_update_magnitude'], 'o-', color='purple')
    axes[1, 1].set_xlabel('Momentum β')
    axes[1, 1].set_ylabel('Mean Update Magnitude')
    axes[1, 1].set_title('Dynamics: Update Size vs β')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(df['beta'], df['final_grad_norm'], 'o-', color='brown')
    axes[1, 2].set_xlabel('Momentum β')
    axes[1, 2].set_ylabel('Final Gradient Norm')
    axes[1, 2].set_title('Convergence: Final Grad Norm vs β')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'momentum_beta_sweep_{test_function}.png', dpi=150)
    plt.close()

    return df


def adam_beta_sweep(
    test_function='rosenbrock',
    beta1_values=np.linspace(0.5, 0.99, 8),
    beta2_values=np.linspace(0.9, 0.9999, 8),
    lr=0.01,
    x0=np.array([-1.5, 2.0]),
    max_iters=1000,
    output_dir='results/hyperparameter_sensitivity'
):
    """
    Systematic 2D sweep of Adam β1, β2 parameters.

    Addresses research proposal requirement: "khảo sát hệ thống ảnh hưởng của
    các siêu tham số đặc trưng (β1, β2 cho Adam)"
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Select test function
    if test_function == 'rosenbrock':
        test_fn = Rosenbrock()
    else:
        test_fn = Ackley2D()

    func = test_fn.compute
    grad_func = lambda x, y: np.array(test_fn.gradient(x, y))

    results = []

    for beta1 in beta1_values:
        for beta2 in beta2_values:
            optimizer = Adam(lr=lr, beta1=beta1, beta2=beta2)

            trajectory, losses, grad_norms = run_optimizer_trajectory(
                optimizer, func, grad_func, x0, max_iters
            )

            # Compute dynamics metrics
            smoothness = compute_trajectory_smoothness(trajectory)
            oscillation = compute_oscillation_index(trajectory)
            final_loss = losses[-1]
            convergence_iters = len(losses)

            # Compute mean update magnitude
            if len(trajectory) > 1:
                updates = np.diff(trajectory, axis=0)
                mean_update_mag = np.mean([np.linalg.norm(u) for u in updates])
            else:
                mean_update_mag = 0.0

            results.append({
                'beta1': beta1,
                'beta2': beta2,
                'final_loss': final_loss,
                'convergence_iters': convergence_iters,
                'smoothness': smoothness,
                'oscillation_index': oscillation,
                'mean_update_magnitude': mean_update_mag,
                'final_grad_norm': grad_norms[-1] if len(grad_norms) > 0 else np.nan
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / f'adam_beta_sweep_{test_function}.csv', index=False)

    # Create 2D heatmaps
    metrics = ['final_loss', 'convergence_iters', 'smoothness', 'oscillation_index']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Pivot for heatmap
        pivot = df.pivot(index='beta2', columns='beta1', values=metric)

        im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(beta1_values)))
        ax.set_yticks(range(len(beta2_values)))
        ax.set_xticklabels([f'{b:.2f}' for b in beta1_values], rotation=45)
        ax.set_yticklabels([f'{b:.4f}' for b in beta2_values])
        ax.set_xlabel('β1 (First Moment Decay)')
        ax.set_ylabel('β2 (Second Moment Decay)')
        ax.set_title(f'Adam: {metric.replace("_", " ").title()}')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'adam_beta_sweep_2d_{test_function}.png', dpi=150)
    plt.close()

    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("Hyperparameter Sensitivity Analysis")
    print("="*60)

    # Momentum β sweep
    print("\n1. Running Momentum β sweep on Rosenbrock...")
    df_mom = momentum_beta_sweep(test_function='rosenbrock')
    print(f"   ✓ Results saved to results/hyperparameter_sensitivity/")

    # Adam β1, β2 sweep
    print("\n2. Running Adam β1, β2 sweep on Rosenbrock...")
    df_adam = adam_beta_sweep(test_function='rosenbrock')
    print(f"   Results saved to results/hyperparameter_sensitivity/")

    print("\nHyperparameter sensitivity analysis complete!")
    print("   - Momentum β: dynamics metrics computed")
    print("   - Adam β1,β2: 2D heatmaps generated")
