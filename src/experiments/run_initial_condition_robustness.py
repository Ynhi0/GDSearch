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

from src.core.test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint
from src.core.optimizers import SGD, SGDMomentum, SGDNesterov, RMSProp, Adam, AdamW, AMSGrad


def run_single_trial(
    optimizer,
    test_function,
    initial_point: Tuple[float, float],
    max_iterations: int,
    convergence_threshold: float = 1e-6
) -> Dict:
    """
    Run one trial with a given initial point.
    
    Returns:
        Dictionary with final_loss, converged (bool), iterations_to_converge, grad_norm
    """
    optimizer.reset()
    x, y = initial_point
    
    for iteration in range(max_iterations):
        loss = test_function.compute(x, y)
        grad_x, grad_y = test_function.gradient(x, y)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        
        # Convergence check
        if grad_norm < convergence_threshold:
            return {
                'final_loss': loss,
                'converged': True,
                'iterations': iteration,
                'grad_norm': grad_norm,
                'final_x': x,
                'final_y': y
            }
        
        x, y = optimizer.step((x, y), (grad_x, grad_y))
    
    # Did not converge within max_iterations
    final_loss = test_function.compute(x, y)
    final_grad_x, final_grad_y = test_function.gradient(x, y)
    final_grad_norm = np.sqrt(final_grad_x**2 + final_grad_y**2)
    
    return {
        'final_loss': final_loss,
        'converged': False,
        'iterations': max_iterations,
        'grad_norm': final_grad_norm,
        'final_x': x,
        'final_y': y
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
                'final_y': trial_result['final_y']
            })
    
    df_detailed = pd.DataFrame(detailed_rows)
    
    # Save detailed results
    detail_path = os.path.join(results_dir, f'initial_condition_robustness_detailed_{func_type}.csv')
    df_detailed.to_csv(detail_path, index=False)
    print(f"\n✅ Detailed results saved to: {detail_path}")
    
    # Aggregate by optimizer
    agg_rows = []
    for opt_name in df_detailed['optimizer'].unique():
        opt_df = df_detailed[df_detailed['optimizer'] == opt_name]
        
        success_rate = opt_df['converged'].mean()
        mean_loss = opt_df['final_loss'].mean()
        std_loss = opt_df['final_loss'].std()
        min_loss = opt_df['final_loss'].min()
        max_loss = opt_df['final_loss'].max()
        
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
            'std_iterations_to_converge': std_iters
        })
    
    df_agg = pd.DataFrame(agg_rows).sort_values('success_rate', ascending=False)
    
    # Save aggregated results
    agg_path = os.path.join(results_dir, f'initial_condition_robustness_summary_{func_type}.csv')
    df_agg.to_csv(agg_path, index=False)
    print(f"✅ Aggregated summary saved to: {agg_path}")
    
    # Plot success rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimizers = df_agg['optimizer'].values
    success_rates = df_agg['success_rate'].values
    
    bars = ax.bar(range(len(optimizers)), success_rates, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(optimizers)))
    ax.set_xticklabels(optimizers, rotation=45, ha='right')
    ax.set_ylabel('Success Rate (Converged / Total Trials)', fontsize=12)
    ax.set_title(f'Initial Condition Robustness: {func_type}\n'
                 f'({len(initial_points)} initial points, convergence threshold={convergence_threshold})',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate bars
    for i, (opt, sr) in enumerate(zip(optimizers, success_rates)):
        ax.text(i, sr + 0.02, f'{sr:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'initial_condition_robustness_{func_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_path}")
    plt.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Initial Condition Robustness Summary: {func_type}")
    print(f"{'='*70}")
    print(df_agg.to_string(index=False))
    print(f"{'='*70}\n")
    
    return df_agg


def main():
    """Example robustness experiment on Rosenbrock."""
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
        results_dir='results',
        plots_dir='plots'
    )
    
    print("\n✅ Robustness experiment complete!")


if __name__ == '__main__':
    main()
