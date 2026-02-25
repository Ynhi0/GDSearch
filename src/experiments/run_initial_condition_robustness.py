"""
2D initial-condition robustness experiments.

Sweeps multiple initial points per optimizer on a 2D function and aggregates outcomes:
- Success rate (converging below threshold)
- Final loss statistics (mean, std, min, max)
- Iteration count statistics
- CSV and plot outputs
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import argparse

from src.core.test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint
from src.core.optimizers import SGD, SGDMomentum, SGDNesterov, RMSProp, Adam, AdamW, AMSGrad
from src.analysis.dynamics_metrics import compute_smoothness_index, compute_oscillation_magnitude


def run_single_trial(
    optimizer,
    test_function,
    initial_point: Tuple[float, float],
    max_iterations: int,
    convergence_threshold: float = 1e-6,
    grad_clip_value: float = 10.0
) -> Dict:
    """
    Run one trial with a given initial point.

    Added gradient clipping to prevent NaN explosion.
    Tracks and returns the optimization trajectory.
    """
    optimizer.reset()
    x, y = initial_point
    trajectory = [(x, y)]
    speeds = []

    for iteration in range(max_iterations):
        loss = test_function.compute(x, y)
        grad_x, grad_y = test_function.gradient(x, y)
        grad_norm = np.hypot(grad_x, grad_y)

        if grad_norm < convergence_threshold:
            return {
                'final_loss': loss,
                'converged': True,
                'iterations': iteration,
                'grad_norm': grad_norm,
                'final_x': x,
                'final_y': y,
                'trajectory': np.array(trajectory),
                'speeds': speeds
            }

        if grad_norm > grad_clip_value:
            clip_scale = grad_clip_value / grad_norm
            grad_x = grad_x * clip_scale
            grad_y = grad_y * clip_scale

        try:
            x_new, y_new = optimizer.step((x, y), (grad_x, grad_y))
            if not (np.isfinite(x_new) and np.isfinite(y_new)):
                return {
                    'final_loss': float('inf'),
                    'converged': False,
                    'iterations': iteration,
                    'grad_norm': float('inf'),
                    'final_x': float('nan'),
                    'final_y': float('nan'),
                    'trajectory': np.array(trajectory),
                    'speeds': speeds
                }
            
            speed = np.hypot(x_new - x, y_new - y)
            speeds.append(speed)
            x, y = x_new, y_new
            trajectory.append((x, y))

        except (ValueError, OverflowError) as e:
            return {
                'final_loss': float('inf'),
                'converged': False,
                'iterations': iteration,
                'grad_norm': grad_norm,
                'final_x': float('nan'),
                'final_y': float('nan'),
                'error': str(e),
                'trajectory': np.array(trajectory),
                'speeds': speeds
            }

    final_loss = test_function.compute(x, y)
    final_grad_x, final_grad_y = test_function.gradient(x, y)
    final_grad_norm = np.hypot(final_grad_x, final_grad_y)

    return {
        'final_loss': final_loss,
        'converged': False,
        'iterations': max_iterations,
        'grad_norm': final_grad_norm,
        'final_x': x,
        'final_y': y,
        'trajectory': np.array(trajectory),
        'speeds': speeds
    }


def generate_initial_points(
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 2.0,
    num_points: int = 20,
    seed: int = 42
) -> List[Tuple[float, float]]:
    """
    Generate initial points in a circle around a center.

    Args:
        center: Center point (x0, y0)
        radius: Radius of circle
        num_points: Number of points to generate
        seed: Random seed

    Returns:
        List of (x, y) tuples
    """
    np.random.seed(seed)
    points = []

    # Use uniform sampling in polar coordinates
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    for angle in angles:
        # Vary radius slightly for diversity
        r = radius * (0.7 + 0.6 * np.random.rand())
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        points.append((x, y))

    return points


def plot_robustness_trajectories(
    test_function,
    detailed_rows: List[Dict],
    func_type: str,
    plots_dir: str
):
    df = pd.DataFrame(detailed_rows)
    optimizers = df['optimizer'].unique()
    
    fig, axes = plt.subplots(int(np.ceil(len(optimizers)/3)), 3, figsize=(18, 6 * int(np.ceil(len(optimizers)/3))))
    if len(optimizers) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    all_x, all_y = [], []
    for row in detailed_rows:
        traj = row.get('trajectory', [])
        if len(traj) > 0:
            all_x.extend(traj[:, 0])
            all_y.extend(traj[:, 1])
    
    if not all_x:
        return
        
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    # Handle zero padding if flat trajectory
    if x_padding == 0: x_padding = 1.0
    if y_padding == 0: y_padding = 1.0
    
    x_range = np.linspace(x_min - x_padding, x_max + x_padding, 100)
    y_range = np.linspace(y_min - y_padding, y_max + y_padding, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[test_function.compute(x, y) for x in x_range] for y in y_range])
    
    for idx, opt in enumerate(optimizers):
        ax = axes[idx]
        opt_df = df[df['optimizer'] == opt]
        
        ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
        
        for _, row in opt_df.iterrows():
            traj = row.get('trajectory', [])
            if len(traj) > 0:
                color = 'r' if row['converged'] else 'gray'
                ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.5, linewidth=1)
                ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=3)
                
        ax.set_title(f'{opt}', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
    for idx in range(len(optimizers), len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'robustness_trajectories_{func_type}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dynamics_metrics(df_agg: pd.DataFrame, func_type: str, plots_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('success_rate', 'Success Rate (Converged / Total)', False),
        ('mean_smoothness', 'Smoothness Index', False),
        ('mean_oscillation', 'Oscillation Metric', False)
    ]
    
    optimizers = df_agg['optimizer'].astype(str).tolist()
    
    for ax, (metric, title, use_log) in zip(axes, metrics):
        if metric not in df_agg.columns:
            continue
        values = df_agg[metric].fillna(0).to_numpy(dtype=float)
        bars = ax.bar(range(len(optimizers)), values, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(optimizers)))
        ax.set_xticklabels(optimizers, rotation=45, ha='right')
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if use_log:
            ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
        
        for i, val in enumerate(values):
            ax.text(i, val + 0.02 * (max(values) if max(values) > 0 else 1.0), 
                    f'{val:.2f}' if metric != 'success_rate' else f'{val:.0%}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'robustness_dynamics_{func_type}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_robustness_experiment(
    optimizer_configs: List[Dict],
    function_config: Dict,
    initial_points: List[Tuple[float, float]],
    max_iterations: int = 5000,
    convergence_threshold: float = 1e-6,
    results_dir: str = 'results',
    plots_dir: str = 'plots'
) -> pd.DataFrame:
    """
    Run robustness experiment across multiple initial points and optimizers.

    Args:
        optimizer_configs: List of dicts with 'type' and 'params'
        function_config: Dict with 'type' and 'params'
        initial_points: List of (x, y) starting points
        max_iterations: Max iterations per trial
        convergence_threshold: Grad norm threshold for convergence
        results_dir: Directory for CSV output
        plots_dir: Directory for plots

    Returns:
        DataFrame with aggregated results
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize test function
    func_type = function_config['type']
    func_params = function_config.get('params', {})

    if func_type == 'Rosenbrock':
        test_function = Rosenbrock(**func_params)
    elif func_type == 'IllConditionedQuadratic':
        test_function = IllConditionedQuadratic(**func_params)
    elif func_type == 'SaddlePoint':
        test_function = SaddlePoint(**func_params)
    else:
        raise ValueError(f"Unknown function type: {func_type}")

    # Collect detailed results
    detailed_rows = []

    for opt_cfg in tqdm(optimizer_configs, desc="Optimizers"):
        opt_type = opt_cfg['type']
        opt_params = opt_cfg.get('params', {})

        # Instantiate optimizer
        if opt_type == 'SGD':
            optimizer = SGD(**opt_params)
        elif opt_type == 'SGDMomentum':
            optimizer = SGDMomentum(**opt_params)
        elif opt_type == 'SGDNesterov':
            optimizer = SGDNesterov(**opt_params)
        elif opt_type == 'RMSProp':
            optimizer = RMSProp(**opt_params)
        elif opt_type == 'Adam':
            optimizer = Adam(**opt_params)
        elif opt_type == 'AdamW':
            optimizer = AdamW(**opt_params)
        elif opt_type == 'AMSGrad':
            optimizer = AMSGrad(**opt_params)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        opt_name = optimizer.name

        # Run trials for all initial points
        for idx, init_pt in enumerate(initial_points):
            trial_result = run_single_trial(
                optimizer, test_function, init_pt, max_iterations, convergence_threshold
            )
            
            traj = trial_result.get('trajectory', np.array([]))
            speeds = trial_result.get('speeds', [])
            mean_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
            
            smoothness = compute_smoothness_index(traj) if len(traj) > 2 else 0.0
            if len(traj) > 2:
                osc_array = compute_oscillation_magnitude(traj)
                oscillation = float(np.mean(osc_array)) if len(osc_array) > 0 else 0.0
            else:
                oscillation = 0.0

            detailed_rows.append({
                'optimizer': opt_name,
                'optimizer_type': opt_type,
                'init_x': init_pt[0],
                'init_y': init_pt[1],
                'init_idx': idx,
                'final_loss': trial_result['final_loss'],
                'converged': trial_result['converged'],
                'iterations': trial_result['iterations'],
                'grad_norm': trial_result['grad_norm'],
                'final_x': trial_result['final_x'],
                'final_y': trial_result['final_y'],
                'mean_speed': mean_speed,
                'smoothness': smoothness,
                'oscillation': oscillation,
                'trajectory': traj
            })

    df_detailed = pd.DataFrame(detailed_rows)

    # Save detailed results, excluding trajectories to save space
    detail_path = os.path.join(results_dir, f'initial_condition_robustness_detailed_{func_type}.csv')
    df_detailed_no_traj = df_detailed.drop(columns=['trajectory']) if 'trajectory' in df_detailed.columns else df_detailed
    df_detailed_no_traj.to_csv(detail_path, index=False)
    print(f"\nDetailed results saved to: {detail_path}")

    # Aggregate by optimizer
    agg_rows = []
    for opt_name in df_detailed['optimizer'].unique():
        opt_df = df_detailed[df_detailed['optimizer'] == opt_name]

        success_rate = opt_df['converged'].mean()
        mean_loss = opt_df['final_loss'].mean()
        std_loss = opt_df['final_loss'].std()
        min_loss = opt_df['final_loss'].min()
        max_loss = opt_df['final_loss'].max()
        
        mean_speed = opt_df['mean_speed'].mean() if 'mean_speed' in opt_df else 0.0
        mean_smoothness = opt_df['smoothness'].mean() if 'smoothness' in opt_df else 0.0
        mean_oscillation = opt_df['oscillation'].mean() if 'oscillation' in opt_df else 0.0

        converged_df = opt_df[opt_df['converged']]
        if len(converged_df) > 0:
            mean_iters = converged_df['iterations'].mean()
            std_iters = converged_df['iterations'].std()
        else:
            mean_iters = np.nan
            std_iters = np.nan

        agg_rows.append({
            'optimizer': opt_name,
            'num_trials': len(opt_df),
            'success_rate': success_rate,
            'mean_final_loss': mean_loss,
            'std_final_loss': std_loss,
            'min_final_loss': min_loss,
            'max_final_loss': max_loss,
            'mean_iterations_to_converge': mean_iters,
            'std_iterations_to_converge': std_iters,
            'mean_speed': mean_speed,
            'mean_smoothness': mean_smoothness,
            'mean_oscillation': mean_oscillation
        })

    from typing import cast
    df_agg = cast(pd.DataFrame, pd.DataFrame(agg_rows)).sort_values(by=['success_rate'], ascending=False)

    # Save aggregated results
    agg_path = os.path.join(results_dir, f'initial_condition_robustness_summary_{func_type}.csv')
    df_agg.to_csv(agg_path, index=False)
    print(f"Aggregated summary saved to: {agg_path}")

    # Plot dynamics and success rates
    plot_robustness_trajectories(test_function, detailed_rows, func_type, plots_dir)
    plot_dynamics_metrics(df_agg, func_type, plots_dir)
    print(f"Plots saved to: {plots_dir}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Initial Condition Robustness Summary: {func_type}")
    print(f"{'='*70}")
    print(df_agg.to_string(index=False))
    print(f"{'='*70}\n")

    return df_agg


def main():
    """Example robustness experiment on Rosenbrock."""
    parser = argparse.ArgumentParser(description='2D Initial Condition Robustness Experiment')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--plots-dir', type=str, default='plots')
    args = parser.parse_args()

    print("="*70)
    print("2D Initial Condition Robustness Experiment")
    print("="*70)

    # Generate initial points around (-1.5, 2.0) - a challenging area for Rosenbrock
    initial_points = generate_initial_points(
        center=(-1.5, 2.0),
        radius=2.5,
        num_points=20,
        seed=42
    )

    print(f"\nGenerated {len(initial_points)} initial points around (-1.5, 2.0)")

    # Define optimizers to test
    optimizer_configs = [
        {'type': 'SGD', 'params': {'lr': 0.001}},
        {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}},
        {'type': 'SGDNesterov', 'params': {'lr': 0.01, 'beta': 0.9}},
        {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9}},
        {'type': 'Adam', 'params': {'lr': 0.01}},
        {'type': 'AdamW', 'params': {'lr': 0.01, 'weight_decay': 0.01}},
        {'type': 'AMSGrad', 'params': {'lr': 0.01}},
    ]

    # Run experiment on Rosenbrock
    function_config = {
        'type': 'Rosenbrock',
        'params': {'a': 1, 'b': 100}
    }

    df_agg = run_robustness_experiment(
        optimizer_configs=optimizer_configs,
        function_config=function_config,
        initial_points=initial_points,
        max_iterations=5000,
        convergence_threshold=1e-6,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir
    )

    print("\nRobustness experiment complete!")


if __name__ == '__main__':
    main()
