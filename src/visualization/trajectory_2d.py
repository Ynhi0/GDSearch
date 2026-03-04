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
    tol: float = 1e-6,
    grad_clip: float | None = None,
    max_param_abs: float = 1e4
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

        if grad_clip is not None and grad_norm > grad_clip:
            grad = grad * (grad_clip / (grad_norm + 1e-12))
            grad_norm = float(np.linalg.norm(grad))

        if grad_norm < tol:
            break

        params = optimizer.step(params, grad)  # type: ignore[attr-defined]
        # Safety guard: if params diverge to non-finite or extremely large values, stop early
        if not np.all(np.isfinite(params)) or np.any(np.abs(params) > max_param_abs):
            logging.warning("Optimizer params diverged (non-finite or large). Stopping trajectory early.")
            break
        trajectory.append(params.copy())
        losses.append(float(func(float(params[0]), float(params[1]))))

    return np.asarray(trajectory, dtype=float), np.asarray(losses, dtype=float)


def _trim_trajectory_for_plot(
    trajectory: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    margin_ratio: float = 0.2,
    max_jump_ratio: float = 0.35,
) -> np.ndarray:
    """
    Keep only the stable, in-bounds prefix of a trajectory.

    This prevents misleading straight-line artifacts when a run takes one
    numerically large jump before divergence handling stops it.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 2 or len(traj) == 0:
        return np.empty((0, 2), dtype=float)

    x_span = max(1e-8, float(xlim[1] - xlim[0]))
    y_span = max(1e-8, float(ylim[1] - ylim[0]))
    x_margin = x_span * margin_ratio
    y_margin = y_span * margin_ratio
    x_min, x_max = xlim[0] - x_margin, xlim[1] + x_margin
    y_min, y_max = ylim[0] - y_margin, ylim[1] + y_margin
    max_jump = max(1e-8, np.hypot(x_span, y_span) * max_jump_ratio)

    out: List[np.ndarray] = [traj[0]]
    for pt in traj[1:]:
        prev = out[-1]
        if not np.all(np.isfinite(pt)):
            break
        if not (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max):
            break
        if float(np.linalg.norm(pt - prev)) > max_jump:
            break
        out.append(pt)

    return np.asarray(out, dtype=float)


from collections.abc import Sequence

def plot_contour_with_trajectories(
    func: Callable[[float, float], float],
    optimizers_config: Sequence[Tuple[str, object, Callable[[float, float], np.ndarray]]],
    x0: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    output_path: Path,
    num_contours: int = 30,
    run_max_iters: int = 500,
    run_tol: float = 1e-6,
    run_grad_clip: float | None = None
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
        trajectory, _ = run_optimizer_2d(
            optimizer,
            func,
            grad_func,
            x0,
            max_iters=run_max_iters,
            tol=run_tol,
            grad_clip=run_grad_clip
        )

        color = colors[idx % len(colors)]
        visible_traj = _trim_trajectory_for_plot(trajectory, xlim, ylim)

        if len(visible_traj) >= 2:
            # Plot stable trajectory line
            ax.plot(visible_traj[:, 0], visible_traj[:, 1],
                   color=color, linewidth=2, alpha=0.7, label=opt_name)
        elif len(visible_traj) == 1:
            ax.plot([], [], color=color, linewidth=2, alpha=0.7, label=opt_name)

        # Plot start point
        ax.plot(x0[0], x0[1], 'o', color=color, markersize=10,
               markeredgecolor='black', markeredgewidth=1.5)

        # Plot end point
        end_pt = visible_traj[-1] if len(visible_traj) > 0 else np.asarray(x0, dtype=float)
        ax.plot(end_pt[0], end_pt[1], '*',
               color=color, markersize=15, markeredgecolor='black', markeredgewidth=1.5)

        # Add arrow annotations to show direction every N steps
        arrow_interval = max(1, len(visible_traj) // 10)
        head_width = 0.015 * (ylim[1] - ylim[0])
        head_length = 0.02 * (xlim[1] - xlim[0])
        for i in range(arrow_interval, len(visible_traj), arrow_interval):
            dx = visible_traj[i, 0] - visible_traj[i-1, 0]
            dy = visible_traj[i, 1] - visible_traj[i-1, 1]
            ax.arrow(visible_traj[i-1, 0], visible_traj[i-1, 1], dx, dy,
                    head_width=head_width, head_length=head_length, fc=color, ec=color, alpha=0.5)

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
    except (OSError, AttributeError):
        # Best-effort: if path isn't a Path object or filesystem error, ignore
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
    Visualize effect of momentum beta on trajectories.

    Notes:
        Accepts either `save_dir` or `output_dir` for backward compatibility with callers.

    Key research question: How does beta shape the trajectory smoothness?
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
        lr_map = {0.0: 1e-3, 0.5: 1e-3, 0.9: 5e-4, 0.99: 2e-4}
        run_max_iters = 5000
        run_grad_clip = 100.0
        title = 'Rosenbrock Function: Momentum beta Effect on Trajectory'
    elif test_function == 'ackley':
        test_fn = Ackley2D()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([2.0, 2.5])
        xlim, ylim = (-5, 5), (-5, 5)
        lr_map = {beta: 0.01 for beta in beta_values}
        run_max_iters = 1200
        run_grad_clip = None
        title = 'Ackley Function: Momentum beta Effect on Trajectory'
    else:
        test_fn = IllConditionedQuadratic()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 2.0])
        xlim, ylim = (-2, 2), (-2, 2)
        lr_map = {beta: 0.01 for beta in beta_values}
        run_max_iters = 1200
        run_grad_clip = None
        title = 'Ill-Conditioned Quadratic: Momentum beta Effect'

    # Build optimizer configs
    optimizers_config = []
    for beta in beta_values:
        lr = float(lr_map.get(beta, 0.01))
        if beta == 0.0:
            opt_name = 'SGD (no momentum)'
            optimizer = SGD(lr=lr)
        else:
            opt_name = f'Momentum beta={beta}'
            optimizer = SGDMomentum(lr=lr, beta=beta)
        optimizers_config.append((opt_name, optimizer, grad_func))

    output_path = Path(save_dir) / f'momentum_beta_trajectories_{test_function}.png'

    plot_contour_with_trajectories(
        func,
        optimizers_config,
        x0,
        xlim,
        ylim,
        title,
        output_path,
        run_max_iters=run_max_iters,
        run_grad_clip=run_grad_clip
    )


def compare_adam_beta_trajectories(
    test_function: str = 'rosenbrock',
    beta_configs: Optional[List[Tuple[float, float]]] = None,
    save_dir: str = 'results/trajectory_visualization',
    output_dir: str | None = None
):
    """
    Visualize effect of Adam beta1, beta2 on trajectories.

    Notes:
        Accepts either `save_dir` or `output_dir` for backward compatibility with callers.

    Key research question: How do beta1, beta2 shape the optimization dynamics?
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
        run_max_iters = 4000
        run_grad_clip = 100.0
        title = 'Rosenbrock Function: Adam beta1,beta2 Effect on Trajectory'
    else:
        test_fn = Ackley2D()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([2.0, 2.5])
        xlim, ylim = (-5, 5), (-5, 5)
        run_max_iters = 1200
        run_grad_clip = None
        title = 'Ackley Function: Adam beta1,beta2 Effect on Trajectory'

    # Build optimizer configs
    optimizers_config = []
    for beta1, beta2 in beta_configs:
        opt_name = f'Adam beta1={beta1}, beta2={beta2}'
        optimizer = Adam(lr=0.01, beta1=beta1, beta2=beta2)
        optimizers_config.append((opt_name, optimizer, grad_func))

    output_path = Path(save_dir) / f'adam_beta_trajectories_{test_function}.png'

    plot_contour_with_trajectories(
        func,
        optimizers_config,
        x0,
        xlim,
        ylim,
        title,
        output_path,
        run_max_iters=run_max_iters,
        run_grad_clip=run_grad_clip
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
        lr_family = 0.001
        run_max_iters = 5000
        run_grad_clip = 100.0
        title = 'Rosenbrock: Optimizer Family Comparison'
    elif test_function == 'saddle':
        test_fn = SaddlePoint()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 1.5])
        xlim, ylim = (-2, 2), (-2, 2)
        lr_family = 0.01
        run_max_iters = 2000
        run_grad_clip = None
        title = 'Saddle Point: Optimizer Family Comparison'
    else:
        test_fn = IllConditionedQuadratic()
        func = test_fn.compute
        grad_func = lambda x, y: np.array(test_fn.gradient(x, y))
        x0 = np.array([-1.5, 2.0])
        xlim, ylim = (-2, 2), (-2, 2)
        lr_family = 0.01
        run_max_iters = 2000
        run_grad_clip = None
        title = 'Ill-Conditioned Quadratic: Optimizer Comparison'

    # Build optimizer configs
    optimizers_config = [
        ('SGD', SGD(lr=lr_family), grad_func),
        ('SGD+Momentum', SGDMomentum(lr=lr_family, beta=0.9), grad_func),
        ('Nesterov', SGDNesterov(lr=lr_family, beta=0.9), grad_func),
        ('Adam', Adam(lr=lr_family), grad_func)
    ]

    output_path = Path(save_dir) / f'optimizer_comparison_{test_function}.png'

    plot_contour_with_trajectories(
        func,
        optimizers_config,
        x0,
        xlim,
        ylim,
        title,
        output_path,
        run_max_iters=run_max_iters,
        run_grad_clip=run_grad_clip
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("2D Trajectory Visualization")
    print("="*60)

    viz_output_dir = 'results/trajectory_visualization'

    # Study 1: Momentum beta effect
    print("\n1. Generating Momentum beta trajectories...")
    compare_momentum_beta_trajectories('rosenbrock', save_dir=viz_output_dir)
    compare_momentum_beta_trajectories('ackley', save_dir=viz_output_dir)

    # Study 2: Adam beta1, beta2 effect
    print("\n2. Generating Adam beta1,beta2 trajectories...")
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
    print("  - Effect of hyperparameters (beta, beta1, beta2)")
    print("  - Optimizer behavior on different landscapes")


