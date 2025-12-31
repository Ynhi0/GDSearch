# -*- coding: utf-8 -*-
""" 2D Trajectory Visualization for Optimizer Dynamics

Creates high-quality visualizations of optimizer trajectories on 2D test functions.
Addresses research proposal requirement: "detailed visualization of kinetic data
(example: 2D trajectory, loss/gradient norm plots over iterations)"

Focus: Visual comparison of dynamics differences between optimizers.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Callable, Optional
import logging

try:
    from src.core.test_functions import (
        Rosenbrock,
        Ackley2D,
        IllConditionedQuadratic,
        SaddlePoint
    )
    from src.core.optimizers import SGD, SGDMomentum, Adam, SGDNesterov
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.test_functions import (
        Rosenbrock,
        Ackley2D,
        IllConditionedQuadratic,
        SaddlePoint
    )
    from src.core.optimizers import SGD, SGDMomentum, Adam, SGDNesterov


def run_optimizer_2d(
    optimizer: object,
    func: Callable[[float, float], float],
    grad_func: Callable[[float, float], np.ndarray],
    x0: np.ndarray,
    max_iters: int = 500,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run optimizer on 2D function and collect trajectory.
    
    Returns:
        trajectory: Array of shape (n_steps, 2) with (x, y) positions
        losses: Array of function values
    """
    from typing import List
    trajectory: List[np.ndarray] = [x0.copy()]
    losses: List[float] = [float(func(*x0))]
    
    params = np.array(x0, dtype=float)
    
    for _ in range(max_iters):
        grad_raw = grad_func(float(params[0]), float(params[1]))
        grad = np.asarray(grad_raw, dtype=float)
        grad_norm = float(np.linalg.norm(grad))
        
        if grad_norm < tol:
            break
        
        params = optimizer.step(params, grad)
        trajectory.append(params.copy())
        losses.append(float(func(float(params[0]), float(params[1]))))
    
    return np.asarray(trajectory, dtype=float), np.asarray(losses, dtype=float)


from collections.abc import Sequence

def plot_contour_with_trajectories(
    func: Callable[[float, float], float],
    optimizers_config: Sequence[Tuple[str, object, Callable[[float, float], np.ndarray]]],
    x0: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    output_path: Path,
    num_contours: int = 30
):
    """
    Create contour plot with multiple optimizer trajectories.
    
    Args:
        func: 2D test function
        optimizers_config: List of (name, optimizer) tuples
        x0: Starting point
        xlim, ylim: Plot bounds
        title: Plot title
        output_path: Where to save the figure
        num_contours: Number of contour lines
    """
    # Create meshgrid for contour plot
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ii = int(i); jj = int(j)
            Z[ii, jj] = float(func(float(X[ii, jj]), float(Y[ii, jj])))
    
    # Create figure
    _, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours
    contour = ax.contour(X, Y, Z, levels=num_contours, cmap='gray', alpha=0.4, linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Plot filled contours for better visualization
    _ = ax.contourf(X, Y, Z, levels=num_contours, cmap='viridis', alpha=0.3)
    
    # Colors for different optimizers
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Plot trajectories
    for idx, (opt_name, optimizer, grad_func) in enumerate(optimizers_config):
        trajectory, _ = run_optimizer_2d(optimizer, func, grad_func, x0)
        
        color = colors[idx % len(colors)]
        
        # Plot trajectory line
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               color=color, linewidth=2, alpha=0.7, label=opt_name)
        
        # Plot start point
        ax.plot(x0[0], x0[1], 'o', color=color, markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5)
        
        # Plot end point
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], '*', 
               color=color, markersize=15, markeredgecolor='black', markeredgewidth=1.5)
        
        # Add arrow annotations to show direction every N steps
        arrow_interval = max(1, len(trajectory) // 10)
        for i in range(arrow_interval, len(trajectory), arrow_interval):
            dx = trajectory[i, 0] - trajectory[i-1, 0]
            dy = trajectory[i, 1] - trajectory[i-1, 1]
            ax.arrow(trajectory[i-1, 0], trajectory[i-1, 1], dx, dy,
                    head_width=0.1, head_length=0.15, fc=color, ec=color, alpha=0.5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info("Saved trajectory plot: %s", output_path)


def plot_vector_field_overlay(
    func: Callable[[float, float], float],
    grad_func: Callable[[float, float], np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: Path,
    density: int = 20,
    normalize: bool = True,
    scale: float | None = None,
    cmap: str = 'plasma'
):
    """
    Plot gradient vector field (quiver) overlaid on contour of the function.

    Args:
        func: 2D function f(x, y)
        grad_func: gradient function returning (gx, gy)
        xlim, ylim: plot bounds
        output_path: Path to save the figure
        density: number of grid points along each axis
        normalize: whether to normalize vectors for consistent lengths
        scale: matplotlib quiver scale parameter
        cmap: colormap for background (magnitude)
    """
    x = np.linspace(xlim[0], xlim[1], density)
    y = np.linspace(ylim[0], ylim[1], density)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    M = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            g = np.asarray(grad_func(float(X[i, j]), float(Y[i, j])), dtype=float)
            gx, gy = float(g[0]), float(g[1])
            U[i, j] = -gx  # plot negative gradient (descent direction)
            V[i, j] = -gy
            M[i, j] = np.sqrt(U[i, j]**2 + V[i, j]**2)

    # Optionally normalize vectors to show direction more clearly
    if normalize:
        nonzero = M > 1e-12
        U[nonzero] = U[nonzero] / M[nonzero]
        V[nonzero] = V[nonzero] / M[nonzero]

    # Create a contour background of function magnitude
    XX = np.linspace(xlim[0], xlim[1], 200)
    YY = np.linspace(ylim[0], ylim[1], 200)
    XXg, YYg = np.meshgrid(XX, YY)
    ZZ = np.zeros_like(XXg, dtype=float)
    for i in range(XXg.shape[0]):
        for j in range(XXg.shape[1]):
            ZZ[i, j] = func(XXg[i, j], YYg[i, j])

    # Plot
    _, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(XXg, YYg, ZZ, levels=30, cmap='viridis', alpha=0.35)
    if normalize:
        # Plot direction-only arrows in a single color
        q = ax.quiver(X, Y, U, V, color='black', scale=scale, alpha=0.8)
    else:
        # Color by magnitude
        q = ax.quiver(X, Y, U, V, M, cmap=cmap, scale=scale)
        plt.colorbar(q, ax=ax, label='Gradient magnitude')
    plt.colorbar(cf, ax=ax, label='Function value')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gradient vector field (descent direction)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Ensure parent directory exists
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort: if path isn't a Path (e.g., string), ignore
        pass
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info("Saved vector field plot: %s", output_path)


def compare_momentum_beta_trajectories(
    test_function: str = 'rosenbrock',
    beta_values: Optional[List[float]] = None,
    save_dir: str = 'results/trajectory_visualization',
    output_dir: str | None = None
):
    """
    Visualize effect of momentum β on trajectories.

    Notes:
        Accepts either `save_dir` or `output_dir` for backward compatibility with callers.
    
    Key research question: How does β shape the trajectory smoothness?
    """
    # Allow callers to pass `output_dir=` (used elsewhere in the codebase)
    if output_dir is not None:
        save_dir = output_dir

    if beta_values is None:
        beta_values = [0.0, 0.5, 0.9, 0.99]
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Select test function
    if test_function == 'rosenbrock':
        test_fn = Rosenbrock()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 2.0])
        xlim, ylim = (-2, 2), (-1, 3)
        title = 'Rosenbrock Function: Momentum β Effect on Trajectory'
    elif test_function == 'ackley':
        test_fn = Ackley2D()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([2.0, 2.5])
        xlim, ylim = (-5, 5), (-5, 5)
        title = 'Ackley Function: Momentum β Effect on Trajectory'
    else:
        test_fn = IllConditionedQuadratic()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 2.0])
        xlim, ylim = (-2, 2), (-2, 2)
        title = 'Ill-Conditioned Quadratic: Momentum β Effect'
    
    # Build optimizer configs
    optimizers_config = []
    for beta in beta_values:
        if beta == 0.0:
            opt_name = 'SGD (no momentum)'
            optimizer = SGD(lr=0.01)
        else:
            opt_name = f'Momentum β={beta}'
            optimizer = SGDMomentum(lr=0.01, beta=beta)
        optimizers_config.append((opt_name, optimizer, grad_func))
    
    output_path = Path(save_dir) / f'momentum_beta_trajectories_{test_function}.png'
    
    plot_contour_with_trajectories(
        func, optimizers_config, x0, xlim, ylim, title, output_path
    )


def compare_adam_beta_trajectories(
    test_function: str = 'rosenbrock',
    beta_configs: Optional[List[Tuple[float, float]]] = None,
    save_dir: str = 'results/trajectory_visualization',
    output_dir: str | None = None
):
    """
    Visualize effect of Adam β1, β2 on trajectories.

    Notes:
        Accepts either `save_dir` or `output_dir` for backward compatibility with callers.

    Key research question: How do β1, β2 shape the optimization dynamics?
    """
    # Allow callers to pass `output_dir=` (used elsewhere in the codebase)
    if output_dir is not None:
        save_dir = output_dir

    if beta_configs is None:
        beta_configs = [(0.9, 0.999), (0.5, 0.99), (0.95, 0.9999)]
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Select test function
    if test_function == 'rosenbrock':
        test_fn = Rosenbrock()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 2.0])
        xlim, ylim = (-2, 2), (-1, 3)
        title = 'Rosenbrock Function: Adam β1,β2 Effect on Trajectory'
    else:
        test_fn = Ackley2D()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([2.0, 2.5])
        xlim, ylim = (-5, 5), (-5, 5)
        title = 'Ackley Function: Adam β1,β2 Effect on Trajectory'
    
    # Build optimizer configs
    optimizers_config = []
    for beta1, beta2 in beta_configs:
        opt_name = f'Adam β1={beta1}, β2={beta2}'
        optimizer = Adam(lr=0.01, beta1=beta1, beta2=beta2)
        optimizers_config.append((opt_name, optimizer, grad_func))
    
    output_path = Path(save_dir) / f'adam_beta_trajectories_{test_function}.png'
    
    plot_contour_with_trajectories(
        func, optimizers_config, x0, xlim, ylim, title, output_path
    )


def compare_optimizer_families(
    test_function: str = 'rosenbrock',
    save_dir: str = 'results/trajectory_visualization',
    output_dir: str | None = None
):
    """
    Compare SGD, Momentum, Nesterov, Adam on same 2D function.

    Notes:
        Accepts either `save_dir` or `output_dir` for backward compatibility with callers.

    Provides side-by-side visual comparison for research presentation.
    """
    # Allow callers to pass `output_dir=` (used elsewhere in the codebase)
    if output_dir is not None:
        save_dir = output_dir

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Select test function
    if test_function == 'rosenbrock':
        test_fn = Rosenbrock()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 2.0])
        xlim, ylim = (-2, 2), (-1, 3)
        title = 'Rosenbrock: Optimizer Family Comparison'
    elif test_function == 'saddle':
        test_fn = SaddlePoint()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 1.5])
        xlim, ylim = (-2, 2), (-2, 2)
        title = 'Saddle Point: Optimizer Family Comparison'
    else:
        test_fn = IllConditionedQuadratic()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 2.0])
        xlim, ylim = (-2, 2), (-2, 2)
        title = 'Ill-Conditioned Quadratic: Optimizer Comparison'
    
    # Build optimizer configs
    optimizers_config = [
        ('SGD', SGD(lr=0.01), grad_func),
        ('SGD+Momentum', SGDMomentum(lr=0.01, beta=0.9), grad_func),
        ('Nesterov', SGDNesterov(lr=0.01, beta=0.9), grad_func),
        ('Adam', Adam(lr=0.01), grad_func)
    ]
    
    output_path = Path(save_dir) / f'optimizer_comparison_{test_function}.png'
    
    plot_contour_with_trajectories(
        func, optimizers_config, x0, xlim, ylim, title, output_path
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("2D Trajectory Visualization")
    print("="*60)
    
    viz_output_dir = 'results/trajectory_visualization'
    
    # Study 1: Momentum β effect
    print("\n1. Generating Momentum β trajectories...")
    compare_momentum_beta_trajectories('rosenbrock', save_dir=viz_output_dir)
    compare_momentum_beta_trajectories('ackley', save_dir=viz_output_dir)
    
    # Study 2: Adam β1, β2 effect
    print("\n2. Generating Adam β1,β2 trajectories...")
    compare_adam_beta_trajectories('rosenbrock', save_dir=viz_output_dir)
    compare_adam_beta_trajectories('ackley', save_dir=viz_output_dir)
    
    # Study 3: Optimizer family comparison
    print("\n3. Generating optimizer family comparisons...")
    compare_optimizer_families('rosenbrock', save_dir=viz_output_dir)
    compare_optimizer_families('saddle', save_dir=viz_output_dir)
    compare_optimizer_families('ill_conditioned', save_dir=viz_output_dir)
    
    print(f"\nAll visualizations saved to {viz_output_dir}/")
    print("\nGenerated plots show:")
    print("  - Trajectory smoothness differences")
    print("  - Effect of hyperparameters (β, β1, β2)")
    print("  - Optimizer behavior on different landscapes")
