"""
Hyperparameter Sensitivity Heatmaps

Visualizes iterations-to-convergence as a 2D heatmap across hyperparameter grids.
Directly addresses the proposal requirement: "systematically survey and visualize 
the influence of characteristic hyperparameters."

Creates heatmaps for:
1. Momentum (β) vs Learning Rate
2. Adam (β1, β2) grid
3. RMSProp (β, ε) grid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from src.core.test_functions import IllConditionedQuadratic, Rosenbrock
from src.core.optimizers import SGDMomentum, Adam, RMSProp
from src.experiments.run_experiment import run_single_experiment


def iterations_to_convergence(
    optimizer_config: Dict,
    function_config: Dict,
    initial_point: Tuple[float, float] = (-1.5, 2.0),
    max_iters: int = 5000,
    loss_threshold: float = 1e-6,
    grad_threshold: float = 1e-5,
    seed: int = 42
) -> int:
    """
    Run single experiment and return iterations to convergence.
    
    Args:
        optimizer_config: Optimizer type and parameters
        function_config: Test function configuration
        initial_point: Starting position
        max_iters: Maximum iterations before declaring non-convergence
        loss_threshold: Convergence criterion for loss
        grad_threshold: Convergence criterion for gradient norm
        seed: Random seed
        
    Returns:
        Number of iterations to convergence (or max_iters if didn't converge)
    """
    df = run_single_experiment(
        optimizer_config=optimizer_config,
        function_config=function_config,
        initial_point=initial_point,
        num_iterations=max_iters,
        seed=seed
    )
    
    # Check convergence using loss threshold
    converged_mask = df['loss'] < loss_threshold
    if converged_mask.any():
        converged_idx = converged_mask.idxmax()  # First True index
        return int(converged_idx)
    
    # If loss didn't converge, check gradient norm
    converged_mask_grad = df['grad_norm'] < grad_threshold
    if converged_mask_grad.any():
        converged_idx = converged_mask_grad.idxmax()
        return int(converged_idx)
    
    # Did not converge
    return max_iters


def momentum_heatmap(
    beta_values: Optional[List[float]] = None,
    lr_values: Optional[List[float]] = None,
    test_function: str = 'ill_conditioned',
    output_dir: str = 'visualizations/heatmaps',
    seed: int = 42
):
    """
    Create 2D heatmap of iterations to convergence for SGD Momentum.
    
    Grid: β (momentum coefficient) vs η (learning rate)
    
    Args:
        beta_values: Momentum values to test
        lr_values: Learning rates to test
        test_function: 'ill_conditioned' or 'rosenbrock'
        output_dir: Directory to save plots
        seed: Random seed
    """
    if beta_values is None:
        beta_values = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    if lr_values is None:
        lr_values = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Select test function
    if test_function == 'ill_conditioned':
        func_config = {'type': 'IllConditionedQuadratic', 'params': {}}
        func_name = 'IllCondQuad'
    else:
        func_config = {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}}
        func_name = 'Rosenbrock'
    
    print(f"Running Momentum heatmap on {func_name}...")
    print(f"  β values: {beta_values}")
    print(f"  η values: {lr_values}")
    
    # Build grid
    grid = np.zeros((len(beta_values), len(lr_values)))
    
    for i, beta in enumerate(beta_values):
        for j, lr in enumerate(lr_values):
            opt_config = {
                'type': 'SGDMomentum',
                'params': {'lr': lr, 'beta': beta}
            }
            
            iters = iterations_to_convergence(
                opt_config, func_config, seed=seed
            )
            grid[i, j] = iters
            
            print(f"  β={beta:.2f}, η={lr:.4f} → {iters} iters")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        grid,
        xticklabels=[f'{lr:.3f}' for lr in lr_values],
        yticklabels=[f'{beta:.2f}' for beta in beta_values],
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',  # Red = slow, Green = fast
        cbar_kws={'label': 'Iterations to Convergence'},
        ax=ax
    )
    
    ax.set_xlabel('Learning Rate (η)', fontsize=12)
    ax.set_ylabel('Momentum Coefficient (β)', fontsize=12)
    ax.set_title(f'SGD Momentum Sensitivity: {func_name}\n(Lower = Faster Convergence)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = Path(output_dir) / f'momentum_heatmap_{func_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved to {save_path}")
    
    # Save data to CSV
    df = pd.DataFrame(
        grid, 
        index=[f'beta_{beta:.2f}' for beta in beta_values],  # type: ignore[arg-type]
        columns=[f'lr_{lr:.4f}' for lr in lr_values]  # type: ignore[arg-type]
    )
    csv_path = Path(output_dir) / f'momentum_heatmap_{func_name}.csv'
    df.to_csv(csv_path)
    print(f"✓ Data saved to {csv_path}")
    
    return grid


def adam_beta_heatmap(
    beta1_values: Optional[List[float]] = None,
    beta2_values: Optional[List[float]] = None,
    lr: float = 0.01,
    test_function: str = 'ill_conditioned',
    output_dir: str = 'visualizations/heatmaps',
    seed: int = 42
):
    """
    Create 2D heatmap for Adam (β1, β2) sensitivity.
    
    Grid: β1 (first moment) vs β2 (second moment)
    
    Args:
        beta1_values: β1 values to test
        beta2_values: β2 values to test
        lr: Fixed learning rate
        test_function: 'ill_conditioned' or 'rosenbrock'
        output_dir: Directory to save plots
        seed: Random seed
    """
    if beta1_values is None:
        beta1_values = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    if beta2_values is None:
        beta2_values = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Select test function
    if test_function == 'ill_conditioned':
        func_config = {'type': 'IllConditionedQuadratic', 'params': {}}
        func_name = 'IllCondQuad'
    else:
        func_config = {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}}
        func_name = 'Rosenbrock'
    
    print(f"Running Adam heatmap on {func_name} (lr={lr})...")
    print(f"  β1 values: {beta1_values}")
    print(f"  β2 values: {beta2_values}")
    
    # Build grid
    grid = np.zeros((len(beta1_values), len(beta2_values)))
    
    for i, beta1 in enumerate(beta1_values):
        for j, beta2 in enumerate(beta2_values):
            opt_config = {
                'type': 'Adam',
                'params': {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'epsilon': 1e-8}
            }
            
            iters = iterations_to_convergence(
                opt_config, func_config, seed=seed
            )
            grid[i, j] = iters
            
            print(f"  β1={beta1:.2f}, β2={beta2:.4f} → {iters} iters")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        grid,
        xticklabels=[f'{b2:.4f}' for b2 in beta2_values],
        yticklabels=[f'{b1:.2f}' for b1 in beta1_values],
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Iterations to Convergence'},
        ax=ax
    )
    
    ax.set_xlabel('β2 (Second Moment Decay)', fontsize=12)
    ax.set_ylabel('β1 (First Moment Decay)', fontsize=12)
    ax.set_title(f'Adam Sensitivity: {func_name} (η={lr})\n(Lower = Faster Convergence)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = Path(output_dir) / f'adam_heatmap_{func_name}_lr{lr}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved to {save_path}")
    
    # Save data
    df = pd.DataFrame(
        grid,
        index=[f'beta1_{b1:.2f}' for b1 in beta1_values],  # type: ignore[arg-type]
        columns=[f'beta2_{b2:.4f}' for b2 in beta2_values]  # type: ignore[arg-type]
    )
    csv_path = Path(output_dir) / f'adam_heatmap_{func_name}_lr{lr}.csv'
    df.to_csv(csv_path)
    print(f"✓ Data saved to {csv_path}")
    
    return grid


def rmsprop_heatmap(
    beta_values: Optional[List[float]] = None,
    epsilon_values: Optional[List[float]] = None,
    lr: float = 0.01,
    test_function: str = 'ill_conditioned',
    output_dir: str = 'visualizations/heatmaps',
    seed: int = 42
):
    """
    Create 2D heatmap for RMSProp (β, ε) sensitivity.
    
    Grid: β (decay rate) vs ε (numerical stability constant)
    
    Args:
        beta_values: β values to test
        epsilon_values: ε values to test
        lr: Fixed learning rate
        test_function: 'ill_conditioned' or 'rosenbrock'
        output_dir: Directory to save plots
        seed: Random seed
    """
    if beta_values is None:
        beta_values = [0.5, 0.7, 0.9, 0.95, 0.99]
    if epsilon_values is None:
        epsilon_values = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Select test function
    if test_function == 'ill_conditioned':
        func_config = {'type': 'IllConditionedQuadratic', 'params': {}}
        func_name = 'IllCondQuad'
    else:
        func_config = {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}}
        func_name = 'Rosenbrock'
    
    print(f"Running RMSProp heatmap on {func_name} (lr={lr})...")
    print(f"  β values: {beta_values}")
    print(f"  ε values: {epsilon_values}")
    
    # Build grid
    grid = np.zeros((len(beta_values), len(epsilon_values)))
    
    for i, beta in enumerate(beta_values):
        for j, epsilon in enumerate(epsilon_values):
            opt_config = {
                'type': 'RMSProp',
                'params': {'lr': lr, 'beta': beta, 'epsilon': epsilon}
            }
            
            iters = iterations_to_convergence(
                opt_config, func_config, seed=seed
            )
            grid[i, j] = iters
            
            print(f"  β={beta:.2f}, ε={epsilon:.1e} → {iters} iters")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        grid,
        xticklabels=[f'{eps:.1e}' for eps in epsilon_values],
        yticklabels=[f'{beta:.2f}' for beta in beta_values],
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Iterations to Convergence'},
        ax=ax
    )
    
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('Beta (β)', fontsize=12)
    ax.set_title(f'RMSProp Sensitivity: {func_name} (η={lr})\n(Lower = Faster Convergence)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = Path(output_dir) / f'rmsprop_heatmap_{func_name}_lr{lr}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved to {save_path}")
    
    # Save data
    df = pd.DataFrame(
        grid,
        index=[f'beta_{beta:.2f}' for beta in beta_values],  # type: ignore[arg-type]
        columns=[f'epsilon_{eps:.1e}' for eps in epsilon_values]  # type: ignore[arg-type]
    )
    csv_path = Path(output_dir) / f'rmsprop_heatmap_{func_name}_lr{lr}.csv'
    df.to_csv(csv_path)
    print(f"✓ Data saved to {csv_path}")
    
    return grid


def run_all_heatmaps(output_dir: str = 'visualizations/heatmaps', seed: int = 42):
    """
    Generate all hyperparameter sensitivity heatmaps.
    
    Creates comprehensive visualizations for:
    - Momentum (β vs η)
    - Adam (β1 vs β2)
    - RMSProp (β vs ε)
    
    On both IllConditionedQuadratic and Rosenbrock test functions.
    """
    print("=" * 60)
    print("Hyperparameter Sensitivity Heatmap Generation")
    print("=" * 60)
    
    # Momentum sensitivity
    print("\n[1/6] Momentum on IllConditionedQuadratic...")
    momentum_heatmap(
        test_function='ill_conditioned',
        output_dir=output_dir,
        seed=seed
    )
    
    print("\n[2/6] Momentum on Rosenbrock...")
    momentum_heatmap(
        test_function='rosenbrock',
        output_dir=output_dir,
        seed=seed
    )
    
    # Adam sensitivity
    print("\n[3/6] Adam on IllConditionedQuadratic...")
    adam_beta_heatmap(
        test_function='ill_conditioned',
        output_dir=output_dir,
        seed=seed
    )
    
    print("\n[4/6] Adam on Rosenbrock...")
    adam_beta_heatmap(
        test_function='rosenbrock',
        output_dir=output_dir,
        seed=seed
    )
    
    # RMSProp sensitivity
    print("\n[5/6] RMSProp on IllConditionedQuadratic...")
    rmsprop_heatmap(
        test_function='ill_conditioned',
        output_dir=output_dir,
        seed=seed
    )
    
    print("\n[6/6] RMSProp on Rosenbrock...")
    rmsprop_heatmap(
        test_function='rosenbrock',
        output_dir=output_dir,
        seed=seed
    )
    
    print("\n" + "=" * 60)
    print("✓ All heatmaps generated successfully!")
    print(f"✓ Results saved to {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate hyperparameter sensitivity heatmaps')
    parser.add_argument('--output-dir', type=str, default='visualizations/heatmaps',
                        help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with fewer grid points')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running in QUICK mode (fewer grid points)")
        # Override with smaller grids for fast testing
        momentum_heatmap(
            beta_values=[0.0, 0.5, 0.9],
            lr_values=[0.01, 0.05, 0.1],
            test_function='ill_conditioned',
            output_dir=args.output_dir,
            seed=args.seed
        )
    else:
        run_all_heatmaps(output_dir=args.output_dir, seed=args.seed)
