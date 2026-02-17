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
    compute_oscillation_metric
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Default configuration for 2D functions
DEFAULT_CONFIGS = {
    'rosenbrock': {
        'initial_point': (-1.5, 2.5),
        'max_iters': 500,
        'lr_sgd': 0.001,
        'lr_momentum': 0.01,
        'lr_adam': 0.1,
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
    },
    'ackley2d': {
        'initial_point': (2.0, 2.0),
        'max_iters': 400,
        'lr_sgd': 0.01,
        'lr_momentum': 0.05,
        'lr_adam': 0.1,
    }
}


def run_single_trial(
    optimizer,
    test_function,
    initial_point: Tuple[float, float],
    max_iters: int
) -> Dict:
    """
    Run single optimization trial and collect dynamics data.
    
    Returns:
        Dictionary with 'trajectory', 'losses', 'grad_norms', 'final_loss'
    """
    params = initial_point
    trajectory = [params]
    losses = [test_function.compute(*params)]
    grad_norms = []
    
    for _ in range(max_iters):
        # Compute gradient
        grad = test_function.gradient(*params)
        grad_norms.append(np.linalg.norm(grad))
        
        # Update parameters
        params = optimizer.step(params, grad)
        trajectory.append(params)
        losses.append(test_function.compute(*params))
        
        # Early stopping if converged
        if grad_norms[-1] < 1e-6:
            break
    
    return {
        'trajectory': np.array(trajectory),
        'losses': np.array(losses),
        'grad_norms': np.array(grad_norms),
        'final_loss': losses[-1],
        'iterations': len(losses) - 1
    }


def compute_dynamics_metrics(result: Dict) -> Dict:
    """Compute dynamics metrics from trial result."""
    trajectory = result['trajectory']
    losses = result['losses']
    
    # Instantaneous speed (step magnitude)
    speeds = []
    for i in range(1, len(trajectory)):
        speed = np.linalg.norm(trajectory[i] - trajectory[i-1])
        speeds.append(speed)
    
    # Smoothness (how consistent are the steps?)
    smoothness = compute_smoothness_index(losses) if len(losses) > 2 else 0.0
    
    # Oscillation (directional changes)
    oscillation = compute_oscillation_metric(trajectory) if len(trajectory) > 2 else 0.0
    
    return {
        'mean_speed': np.mean(speeds) if speeds else 0.0,
        'std_speed': np.std(speeds) if speeds else 0.0,
        'smoothness': smoothness,
        'oscillation': oscillation,
        'final_grad_norm': result['grad_norms'][-1] if result['grad_norms'] else np.inf
    }


def run_beta_sweep_momentum(
    test_function,
    beta_values: List[float],
    lr: float,
    initial_point: Tuple[float, float],
    max_iters: int,
    output_dir: Path
) -> pd.DataFrame:
    """Run Momentum optimizer with different β values."""
    logger.info(f"Running Momentum beta sweep: β ∈ {beta_values}")
    
    results = []
    all_trajectories = {}
    
    for beta in tqdm(beta_values, desc="Beta sweep"):
        optimizer = SGDMomentum(lr=lr, beta=beta)
        trial_result = run_single_trial(optimizer, test_function, initial_point, max_iters)
        dynamics = compute_dynamics_metrics(trial_result)
        
        results.append({
            'beta': beta,
            'final_loss': trial_result['final_loss'],
            'iterations': trial_result['iterations'],
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
    output_dir: Path
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
                trial_result = run_single_trial(optimizer, test_function, initial_point, max_iters)
                dynamics = compute_dynamics_metrics(trial_result)
                
                results.append({
                    'beta1': beta1,
                    'beta2': beta2,
                    'final_loss': trial_result['final_loss'],
                    'iterations': trial_result['iterations'],
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
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[test_function.compute(x, y) for x in x_range] for y in y_range])
    
    for idx, beta in enumerate(beta_values[:6]):  # Max 6 subplots
        ax = axes[idx]
        
        # Contour background
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Trajectory
        traj = trajectories[beta]['trajectory']
        ax.plot(traj[:, 0], traj[:, 1], 'r.-', linewidth=2, markersize=4, label=f'β={beta}')
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End')
        
        ax.set_title(f'Momentum β={beta}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
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
        ax.plot(df['beta'], df[metric], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('β (Momentum Coefficient)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs β', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log' if metric in ['final_loss', 'final_grad_norm'] else 'linear')
    
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
            test_function, beta_values, lr, initial_point, max_iters, output_dir
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
            test_function, beta1_values, beta2_values, lr, initial_point, max_iters, output_dir
        )
        
        # Visualizations
        plot_adam_heatmaps(df, output_dir)
        
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"\n{df.to_string()}")
    
    logger.info(f"\n✅ Analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
