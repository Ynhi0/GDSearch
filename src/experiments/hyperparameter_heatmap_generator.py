"""
Hyperparameter Sensitivity Heatmap Generator

Creates 2D heatmaps visualizing the influence of optimizer hyperparameters
on convergence speed. Fulfills research proposal requirement:
"systematically survey and visualize the influence of characteristic hyperparameters"

Generates:
1. Momentum Beta Grid: β vs iterations-to-converge
2. Adam Beta Grid: β₁ vs β₂ vs iterations-to-converge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from src.core.test_functions import Rosenbrock, IllConditionedQuadratic
from src.core.optimizers import SGDMomentum, Adam

logging.basicConfig(level=logging.INFO)


def run_momentum_beta_heatmap(
    test_function=None,
    initial_point: Tuple[float, float] = (-1.5, 2.0),
    beta_range: Optional[List[float]] = None,
    lr_range: Optional[List[float]] = None,
    max_iters: int = 5000,
    convergence_threshold: float = 1e-6,
    output_dir: str = 'results/hyperparameter_heatmaps'
) -> pd.DataFrame:
    """
    Generate heatmap of Momentum beta vs learning rate vs convergence speed.
    
    Args:
        test_function: Test function (default: Rosenbrock)
        initial_point: Starting point
        beta_range: List of beta values to test
        lr_range: List of learning rates to test
        max_iters: Maximum iterations before timeout
        convergence_threshold: Gradient norm threshold for convergence
        output_dir: Directory to save results
        
    Returns:
        DataFrame with beta, lr, iterations_to_converge columns
    """
    if test_function is None:
        test_function = Rosenbrock()
    
    # Set defaults and convert to list for type safety
    beta_list: List[float] = list(beta_range) if beta_range is not None else list(np.linspace(0.0, 0.99, 20))
    lr_list: List[float] = list(lr_range) if lr_range is not None else list(np.logspace(-4, -1, 15))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print("\n" + "="*60)
    print("Momentum Beta Sensitivity Heatmap")
    print("="*60)
    print(f"Test Function: {test_function.__class__.__name__}")
    print(f"Beta range: [{beta_list[0]:.2f}, {beta_list[-1]:.2f}] ({len(beta_list)} values)")
    print(f"LR range: [{lr_list[0]:.4f}, {lr_list[-1]:.4f}] ({len(lr_list)} values)")
    print(f"Total configurations: {len(beta_list) * len(lr_list)}")
    
    total = len(beta_list) * len(lr_list)
    count = 0
    
    for beta in beta_list:
        for lr in lr_list:
            count += 1
            if count % 20 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
            
            optimizer = SGDMomentum(lr=lr, beta=beta)
            x, y = initial_point
            
            converged = False
            iters = 0
            
            for i in range(max_iters):
                grad_x, grad_y = test_function.gradient(x, y)
                grad_norm = np.linalg.norm([grad_x, grad_y])
                
                if not np.isfinite(grad_norm):
                    # Diverged
                    iters = max_iters
                    break
                
                if grad_norm < convergence_threshold:
                    converged = True
                    iters = i
                    break
                
                x, y = optimizer.step((x, y), (grad_x, grad_y))
                
                if not np.isfinite(x) or not np.isfinite(y):
                    # Diverged
                    iters = max_iters
                    break
            
            if not converged:
                iters = max_iters
            
            results.append({
                'beta': beta,
                'lr': lr,
                'iterations_to_converge': iters,
                'converged': converged
            })
    
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'momentum_beta_heatmap_data.csv', index=False)
    
    # Generate heatmap
    pivot = df.pivot(index='lr', columns='beta', values='iterations_to_converge')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap='viridis_r', annot=False, fmt='.0f', cbar_kws={'label': 'Iterations to Converge'})
    plt.xlabel('Momentum Beta (β)', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(f'Momentum Hyperparameter Sensitivity\n{test_function.__class__.__name__}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'momentum_beta_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nHeatmap saved to {output_dir}/momentum_beta_heatmap.png")
    print(f"Data saved to {output_dir}/momentum_beta_heatmap_data.csv")
    
    return df


def run_adam_beta_heatmap(
    test_function=None,
    initial_point: Tuple[float, float] = (-1.5, 2.0),
    beta1_range: Optional[List[float]] = None,
    beta2_range: Optional[List[float]] = None,
    lr: float = 0.001,
    max_iters: int = 5000,
    convergence_threshold: float = 1e-6,
    output_dir: str = 'results/hyperparameter_heatmaps'
) -> pd.DataFrame:
    """
    Generate heatmap of Adam beta1 vs beta2 vs convergence speed.
    
    Args:
        test_function: Test function (default: Rosenbrock)
        initial_point: Starting point
        beta1_range: List of beta1 values to test
        beta2_range: List of beta2 values to test
        lr: Fixed learning rate
        max_iters: Maximum iterations before timeout
        convergence_threshold: Gradient norm threshold for convergence
        output_dir: Directory to save results
        
    Returns:
        DataFrame with beta1, beta2, iterations_to_converge columns
    """
    if test_function is None:
        test_function = Rosenbrock()
    
    # Convert to lists for type safety
    beta1_values: List[float] = list(beta1_range) if beta1_range is not None else list(np.linspace(0.5, 0.99, 15))
    beta2_values: List[float] = list(beta2_range) if beta2_range is not None else list(np.linspace(0.9, 0.9999, 15))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print("\n" + "="*60)
    print("Adam Beta Sensitivity Heatmap")
    print("="*60)
    print(f"Test Function: {test_function.__class__.__name__}")
    print(f"Beta1 range: [{beta1_values[0]:.3f}, {beta1_values[-1]:.3f}] ({len(beta1_values)} values)")
    print(f"Beta2 range: [{beta2_values[0]:.3f}, {beta2_values[-1]:.4f}] ({len(beta2_values)} values)")
    print(f"Fixed LR: {lr}")
    print(f"Total configurations: {len(beta1_values) * len(beta2_values)}")
    
    total = len(beta1_values) * len(beta2_values)
    count = 0
    
    for beta1 in beta1_values:
        for beta2 in beta2_values:
            count += 1
            if count % 20 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
            
            optimizer = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=1e-8)
            x, y = initial_point
            
            converged = False
            iters = 0
            
            for i in range(max_iters):
                grad_x, grad_y = test_function.gradient(x, y)
                grad_norm = np.linalg.norm([grad_x, grad_y])
                
                if not np.isfinite(grad_norm):
                    iters = max_iters
                    break
                
                if grad_norm < convergence_threshold:
                    converged = True
                    iters = i
                    break
                
                x, y = optimizer.step((x, y), (grad_x, grad_y))
                
                if not np.isfinite(x) or not np.isfinite(y):
                    iters = max_iters
                    break
            
            if not converged:
                iters = max_iters
            
            results.append({
                'beta1': beta1,
                'beta2': beta2,
                'iterations_to_converge': iters,
                'converged': converged
            })
    
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'adam_beta_heatmap_data.csv', index=False)
    
    # Generate heatmap
    pivot = df.pivot(index='beta2', columns='beta1', values='iterations_to_converge')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap='viridis_r', annot=False, fmt='.0f', cbar_kws={'label': 'Iterations to Converge'})
    plt.xlabel('Beta1 (β₁) - First Moment', fontsize=12)
    plt.ylabel('Beta2 (β₂) - Second Moment', fontsize=12)
    plt.title(f'Adam Hyperparameter Sensitivity\n{test_function.__class__.__name__} (lr={lr})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'adam_beta_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nHeatmap saved to {output_dir}/adam_beta_heatmap.png")
    print(f"Data saved to {output_dir}/adam_beta_heatmap_data.csv")
    
    return df


if __name__ == '__main__':
    # Generate both heatmaps
    momentum_df = run_momentum_beta_heatmap(
        test_function=Rosenbrock(),
        output_dir='results/hyperparameter_heatmaps'
    )
    
    adam_df = run_adam_beta_heatmap(
        test_function=Rosenbrock(),
        lr=0.001,
        output_dir='results/hyperparameter_heatmaps'
    )
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS COMPLETE")
    print("="*60)
    print("Results address research proposal objective:")
    print("'Systematically survey and visualize the influence of characteristic hyperparameters'")
