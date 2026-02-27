#!/usr/bin/env python3
"""
Beta Parameter Sensitivity Analysis on 2D Test Functions

Complementary to beta_sensitivity_training.py (which runs on neural networks),
this script analyzes β, β1, β2 impact on 2D test function optimization.

Research Proposal Alignment:
"khảo sát hệ thống và trực quan hóa ảnh hưởng của các siêu tham số đặc trưng
(β, β1, β2) lên các khía cạnh động học như quỹ đạo, tốc độ tức thời và độ ổn định"

Purpose:
- Generate clear 2D trajectory visualizations for thesis figures
- Analyze β impact on Momentum optimizer dynamics
- Analyze β1, β2 impact on Adam optimizer dynamics
- Compare convergence speed, smoothness, and stability across β values

Output:
- Trajectory plots (contour + optimizer path)
- Convergence rate comparisons
- Dynamics metrics (smoothness, oscillation, instantaneous speed)

Usage:
    # Momentum beta sweep on Rosenbrock
    python src/experiments/beta_sensitivity_2d.py --optimizer Momentum \\
        --function rosenbrock --beta-values 0.5,0.7,0.9,0.95,0.99
    
    # Adam beta1/beta2 sweep on Saddle Point
    python src/experiments/beta_sensitivity_2d.py --optimizer Adam \\
        --function saddle_point --beta1-values 0.8,0.9,0.95 \\
        --beta2-values 0.9,0.99,0.999

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from tqdm import tqdm

from src.core.test_function_factory import get_test_function
from src.core.optimizers import SGD, SGDMomentum, SGDNesterov, Adam, AdamW, RMSProp
from src.analysis.dynamics_metrics import (
    compute_instantaneous_speed, compute_smoothness_index,
    compute_oscillation_magnitude
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Default configuration for 2D functions
DEFAULT_CONFIGS = {
    'rosenbrock': {
        'initial_point': (-1.5, 2.5),
        'max_iters': 1500,
        'lr_sgd': 0.001,
        'lr_momentum': 0.001,
        'lr_adam': 0.01,
        'grad_clip': 1000.0,
        'max_param_abs': 50.0,
        'loss_cap': 1e8,
    },
    'ill_conditioned_quadratic': {
        'initial_point': (3.0, 3.0),
        'max_iters': 300,
        'lr_sgd': 0.01,
        'lr_momentum': 0.05,
        'lr_adam': 0.2,
    },
    'saddle_point': {
        'initial_point': (1.0, 1.0),
        'max_iters': 200,
        'lr_sgd': 0.05,
        'lr_momentum': 0.1,
        'lr_adam': 0.2,
        'grad_clip': 1000.0,
        'max_param_abs': 100.0,
        'loss_cap': 1e8,
    },
    'ackley2d': {
        'initial_point': (2.0, 2.0),
        'max_iters': 400,
        'lr_sgd': 0.01,
        'lr_momentum': 0.05,
        'lr_adam': 0.1,
        'grad_clip': 1000.0,
        'max_param_abs': 100.0,
        'loss_cap': 1e8,
    }
}


def _sanitize_metric(value: float, max_abs: float = 1e12) -> float:
    """Convert non-finite/exploded values to NaN for robust CSV + plotting."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(v):
        return float('nan')
    if abs(v) > max_abs:
        return float('nan')
    return v


def _trajectory_prefix_within_bounds(
    trajectory: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    margin_ratio: float = 0.1
) -> np.ndarray:
    """Return finite in-bounds prefix to avoid plotting exploded jumps."""
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 2 or len(traj) == 0:
        return np.empty((0, 2), dtype=float)

    x_margin = (xlim[1] - xlim[0]) * margin_ratio
    y_margin = (ylim[1] - ylim[0]) * margin_ratio
    x_min, x_max = xlim[0] - x_margin, xlim[1] + x_margin
    y_min, y_max = ylim[0] - y_margin, ylim[1] + y_margin

    valid_prefix = []
    for pt in traj:
        if not np.all(np.isfinite(pt)):
            break
        if not (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max):
            break
        valid_prefix.append(pt)

    if not valid_prefix:
        return np.empty((0, 2), dtype=float)
    return np.asarray(valid_prefix, dtype=float)


def run_single_trial(
    optimizer,
    test_function,
    initial_point: Tuple[float, float],
    max_iters: int,
    grad_clip: Optional[float] = None,
    max_param_abs: float = 1e4,
    loss_cap: float = 1e12,
    tol: float = 1e-6,
) -> Dict:
    """
    Run single optimization trial and collect dynamics data.
    
    Returns:
        Dictionary with 'trajectory', 'losses', 'grad_norms', 'final_loss'
    """
    params = np.asarray(initial_point, dtype=float)
    trajectory = [params.copy()]
    losses = [float(test_function.compute(float(params[0]), float(params[1])))]
    grad_norms = []
    status = 'max_iters'

    for _ in range(max_iters):
        try:
            # Compute gradient
            grad = np.asarray(
                test_function.gradient(float(params[0]), float(params[1])),
                dtype=float
            )

            # Check for NaN/Inf
            if not np.all(np.isfinite(grad)):
                status = 'non_finite_gradient'
                break

            grad_norm = float(np.linalg.norm(grad))
            if grad_clip is not None and grad_norm > grad_clip:
                grad = grad * (grad_clip / (grad_norm + 1e-12))
                grad_norm = float(np.linalg.norm(grad))
            grad_norms.append(grad_norm)

            # Early stopping if converged
            if grad_norm < tol:
                status = 'converged'
                break

            # Update parameters
            next_params = np.asarray(optimizer.step(params, grad), dtype=float)
            if next_params.shape != (2,):
                status = 'invalid_param_shape'
                break
            if not np.all(np.isfinite(next_params)):
                status = 'non_finite_params'
                break

            if np.any(np.abs(next_params) > max_param_abs):
                status = 'params_out_of_bounds'
                break

            next_loss = float(test_function.compute(float(next_params[0]), float(next_params[1])))
            if not np.isfinite(next_loss):
                status = 'non_finite_loss'
                break
            if abs(next_loss) > loss_cap:
                status = 'loss_exploded'
                break

            params = next_params
            trajectory.append(params.copy())
            losses.append(next_loss)
        except (OverflowError, ValueError, TypeError):
            status = 'numeric_exception'
            break

    diverged_statuses = {
        'non_finite_gradient',
        'non_finite_params',
        'params_out_of_bounds',
        'non_finite_loss',
        'loss_exploded',
        'numeric_exception',
        'invalid_param_shape',
    }
    return {
        'trajectory': np.array(trajectory),
        'losses': np.array(losses),
        'grad_norms': np.array(grad_norms),
        'final_loss': _sanitize_metric(losses[-1], max_abs=loss_cap),
        'iterations': len(losses) - 1,
        'converged': status == 'converged',
        'diverged': status in diverged_statuses,
        'status': status,
    }


def compute_dynamics_metrics(result: Dict) -> Dict:
    """Compute dynamics metrics from trial result."""
    trajectory = result['trajectory']

    # Instantaneous speed (step magnitude)
    if len(trajectory) > 1:
        step_sizes = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        finite_step_sizes = step_sizes[np.isfinite(step_sizes)]
    else:
        finite_step_sizes = np.array([], dtype=float)

    # Smoothness (how consistent are the steps?)
    if len(trajectory) > 2 and np.all(np.isfinite(trajectory)):
        try:
            smoothness = _sanitize_metric(compute_smoothness_index(trajectory))
        except (OverflowError, ValueError, TypeError):
            smoothness = float('nan')
    else:
        smoothness = float('nan')

    # Oscillation (directional changes)
    if len(trajectory) > 2 and np.all(np.isfinite(trajectory)):
        osc_array = np.asarray(compute_oscillation_magnitude(trajectory), dtype=float)
        osc_array = osc_array[np.isfinite(osc_array)]
        oscillation = float(np.mean(osc_array)) if len(osc_array) > 0 else 0.0
    else:
        oscillation = float('nan')

    final_grad_norm = float('nan')
    if len(result['grad_norms']) > 0:
        final_grad_norm = _sanitize_metric(float(result['grad_norms'][-1]))

    return {
        'mean_speed': _sanitize_metric(np.mean(finite_step_sizes)) if len(finite_step_sizes) > 0 else float('nan'),
        'std_speed': _sanitize_metric(np.std(finite_step_sizes)) if len(finite_step_sizes) > 0 else float('nan'),
        'smoothness': smoothness,
        'oscillation': _sanitize_metric(oscillation),
        'final_grad_norm': final_grad_norm
    }


def run_beta_sweep_momentum(
    test_function,
    beta_values: List[float],
    lr: float,
    initial_point: Tuple[float, float],
    max_iters: int,
    output_dir: Path,
    grad_clip: Optional[float] = None,
    max_param_abs: float = 1e4,
    loss_cap: float = 1e12,
) -> pd.DataFrame:
    """Run Momentum optimizer with different β values."""
    logger.info(f"Running Momentum beta sweep: β ∈ {beta_values}")
    
    results = []
    all_trajectories = {}
    
    for beta in tqdm(beta_values, desc="Beta sweep"):
        optimizer = SGDMomentum(lr=lr, beta=beta)
        trial_result = run_single_trial(
            optimizer,
            test_function,
            initial_point,
            max_iters,
            grad_clip=grad_clip,
            max_param_abs=max_param_abs,
            loss_cap=loss_cap,
        )
        dynamics = compute_dynamics_metrics(trial_result)
        
        results.append({
            'beta': beta,
            'final_loss': trial_result['final_loss'],
            'iterations': trial_result['iterations'],
            'converged': trial_result['converged'],
            'diverged': trial_result['diverged'],
            'status': trial_result['status'],
            'mean_speed': dynamics['mean_speed'],
            'std_speed': dynamics['std_speed'],
            'smoothness': dynamics['smoothness'],
            'oscillation': dynamics['oscillation'],
            'final_grad_norm': dynamics['final_grad_norm']
        })
        
        all_trajectories[beta] = trial_result
    
    # Save trajectories for visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'momentum_trajectories.npy', all_trajectories)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'momentum_beta_sweep.csv', index=False)
    logger.info(f"Saved results to {output_dir / 'momentum_beta_sweep.csv'}")
    
    return df, all_trajectories


def run_beta_sweep_adam(
    test_function,
    beta1_values: List[float],
    beta2_values: List[float],
    lr: float,
    initial_point: Tuple[float, float],
    max_iters: int,
    output_dir: Path,
    grad_clip: Optional[float] = None,
    max_param_abs: float = 1e4,
    loss_cap: float = 1e12,
) -> pd.DataFrame:
    """Run Adam optimizer with different β1, β2 combinations."""
    logger.info(f"Running Adam beta sweep: β1 ∈ {beta1_values}, β2 ∈ {beta2_values}")
    
    results = []
    all_trajectories = {}
    
    total = len(beta1_values) * len(beta2_values)
    with tqdm(total=total, desc="Adam β1×β2 sweep") as pbar:
        for beta1 in beta1_values:
            for beta2 in beta2_values:
                optimizer = Adam(lr=lr, beta1=beta1, beta2=beta2)
                trial_result = run_single_trial(
                    optimizer,
                    test_function,
                    initial_point,
                    max_iters,
                    grad_clip=grad_clip,
                    max_param_abs=max_param_abs,
                    loss_cap=loss_cap,
                )
                dynamics = compute_dynamics_metrics(trial_result)
                
                results.append({
                    'beta1': beta1,
                    'beta2': beta2,
                    'final_loss': trial_result['final_loss'],
                    'iterations': trial_result['iterations'],
                    'converged': trial_result['converged'],
                    'diverged': trial_result['diverged'],
                    'status': trial_result['status'],
                    'mean_speed': dynamics['mean_speed'],
                    'std_speed': dynamics['std_speed'],
                    'smoothness': dynamics['smoothness'],
                    'oscillation': dynamics['oscillation'],
                    'final_grad_norm': dynamics['final_grad_norm']
                })
                
                all_trajectories[(beta1, beta2)] = trial_result
                pbar.update(1)
    
    # Save trajectories
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'adam_trajectories.npy', all_trajectories)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'adam_beta_sweep.csv', index=False)
    logger.info(f"Saved results to {output_dir / 'adam_beta_sweep.csv'}")
    
    return df, all_trajectories


def plot_momentum_trajectories(
    test_function,
    trajectories: Dict,
    beta_values: List[float],
    output_dir: Path
):
    """Plot 2D trajectories for different β values (Momentum)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Create contour plot
    try:
        xlim, ylim = test_function.get_bounds()
    except AttributeError:
        xlim, ylim = (-2, 2), (-1, 3)
    x_range = np.linspace(xlim[0], xlim[1], 100)
    y_range = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[test_function.compute(x, y) for x in x_range] for y in y_range])
    
    for idx, beta in enumerate(beta_values[:6]):  # Max 6 subplots
        ax = axes[idx]
        
        # Contour background
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Trajectory
        traj = np.asarray(trajectories[beta]['trajectory'], dtype=float)
        visible_traj = _trajectory_prefix_within_bounds(traj, xlim, ylim)
        status = str(trajectories[beta].get('status', 'unknown'))

        if len(visible_traj) >= 2:
            ax.plot(visible_traj[:, 0], visible_traj[:, 1], 'r.-', linewidth=2, markersize=4, label=f'β={beta}')
            ax.plot(visible_traj[0, 0], visible_traj[0, 1], 'go', markersize=10, label='Start')
            ax.plot(visible_traj[-1, 0], visible_traj[-1, 1], 'r*', markersize=15, label='End')
        elif len(traj) > 0 and np.all(np.isfinite(traj[0])):
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
            ax.text(0.02, 0.9, 'No visible stable trajectory', transform=ax.transAxes, fontsize=10, color='darkred')

        if status != 'converged':
            ax.text(
                0.02, 0.02, f'status: {status}', transform=ax.transAxes,
                fontsize=9, color='darkred',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        
        ax.set_title(f'Momentum β={beta}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(beta_values), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'momentum_trajectories.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved trajectory plot to {output_dir / 'momentum_trajectories.png'}")
    plt.close()


def plot_momentum_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot dynamics metrics vs β for Momentum."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = [
        ('final_loss', 'Final Loss'),
        ('iterations', 'Iterations to Convergence'),
        ('mean_speed', 'Mean Step Size'),
        ('smoothness', 'Smoothness Index'),
        ('oscillation', 'Oscillation Metric'),
        ('final_grad_norm', 'Final Gradient Norm')
    ]
    
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        y = pd.to_numeric(df[metric], errors='coerce')
        finite_mask = np.isfinite(y.to_numpy())
        if metric in ['final_loss', 'final_grad_norm']:
            finite_mask = finite_mask & (y.to_numpy() > 0)

        if finite_mask.any():
            ax.plot(df.loc[finite_mask, 'beta'], y[finite_mask], 'o-', linewidth=2, markersize=8)
        else:
            ax.text(0.5, 0.5, 'No finite data', transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='darkred')

        invalid_count = int((~finite_mask).sum())
        if invalid_count > 0:
            ax.text(0.98, 0.02, f'invalid={invalid_count}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9, color='darkred')

        ax.set_xlabel('β (Momentum Coefficient)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs β', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if metric in ['final_loss', 'final_grad_norm'] and finite_mask.any():
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'momentum_metrics.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved metrics plot to {output_dir / 'momentum_metrics.png'}")
    plt.close()


def plot_adam_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Plot heatmaps showing β1 vs β2 impact on various metrics."""
    metrics = ['final_loss', 'iterations', 'smoothness', 'oscillation']
    titles = ['Final Loss', 'Iterations', 'Smoothness', 'Oscillation']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, metric, title in zip(axes, metrics, titles):
        # Pivot for heatmap
        heatmap_data = df.pivot(index='beta2', columns='beta1', values=metric)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r' if metric in ['final_loss', 'oscillation'] else 'RdYlGn',
                    ax=ax, cbar_kws={'label': title})
        ax.set_title(f'{title} (β1 vs β2)', fontsize=14, fontweight='bold')
        ax.set_xlabel('β1 (First Moment Decay)', fontsize=12)
        ax.set_ylabel('β2 (Second Moment Decay)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'adam_heatmaps.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved Adam heatmaps to {output_dir / 'adam_heatmaps.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Beta sensitivity analysis on 2D test functions')
    parser.add_argument('--optimizer', type=str, required=True, choices=['Momentum', 'Adam'],
                        help='Optimizer to analyze')
    parser.add_argument('--function', type=str, required=True,
                        choices=['rosenbrock', 'ill_conditioned_quadratic', 'saddle_point', 'ackley2d'],
                        help='Test function to optimize')
    parser.add_argument('--beta-values', type=str, default='0.5,0.7,0.9,0.95,0.99',
                        help='Comma-separated beta values for Momentum (default: 0.5,0.7,0.9,0.95,0.99)')
    parser.add_argument('--beta1-values', type=str, default='0.8,0.9,0.95',
                        help='Comma-separated beta1 values for Adam (default: 0.8,0.9,0.95)')
    parser.add_argument('--beta2-values', type=str, default='0.9,0.99,0.999',
                        help='Comma-separated beta2 values for Adam (default: 0.9,0.99,0.999)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: function-specific)')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='Maximum iterations (default: function-specific)')
    parser.add_argument('--output-dir', type=str, default='results/beta_sensitivity_2d',
                        help='Output directory (default: results/beta_sensitivity_2d)')
    
    args = parser.parse_args()
    
    # Load test function
    test_function = get_test_function(args.function)
    config = DEFAULT_CONFIGS[args.function]
    
    # Override defaults if provided
    lr = args.lr if args.lr is not None else config[f'lr_{args.optimizer.lower()}']
    max_iters = args.max_iters if args.max_iters is not None else config['max_iters']
    initial_point = config['initial_point']
    grad_clip = config.get('grad_clip')
    max_param_abs = float(config.get('max_param_abs', 1e4))
    loss_cap = float(config.get('loss_cap', 1e12))
    
    output_dir = Path(args.output_dir) / args.function / args.optimizer.lower()
    
    logger.info(f"=== Beta Sensitivity Analysis ===")
    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"Function: {args.function}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Max iterations: {max_iters}")
    logger.info(f"Initial point: {initial_point}")
    
    if args.optimizer == 'Momentum':
        beta_values = [float(x) for x in args.beta_values.split(',')]
        df, trajectories = run_beta_sweep_momentum(
            test_function,
            beta_values,
            lr,
            initial_point,
            max_iters,
            output_dir,
            grad_clip=grad_clip,
            max_param_abs=max_param_abs,
            loss_cap=loss_cap,
        )
        
        # Visualizations
        plot_momentum_trajectories(test_function, trajectories, beta_values, output_dir)
        plot_momentum_metrics(df, output_dir)
        
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"\n{df.to_string()}")
        
    elif args.optimizer == 'Adam':
        beta1_values = [float(x) for x in args.beta1_values.split(',')]
        beta2_values = [float(x) for x in args.beta2_values.split(',')]
        df, trajectories = run_beta_sweep_adam(
            test_function,
            beta1_values,
            beta2_values,
            lr,
            initial_point,
            max_iters,
            output_dir,
            grad_clip=grad_clip,
            max_param_abs=max_param_abs,
            loss_cap=loss_cap,
        )
        
        # Visualizations
        plot_adam_heatmaps(df, output_dir)
        
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"\n{df.to_string()}")
    
    logger.info(f"\n[OK] Analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
