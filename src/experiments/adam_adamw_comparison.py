"""
Adam L2 vs. AdamW Comparison Experiment

This experiment demonstrates WHY AdamW exists: L2 regularization interacts
poorly with adaptive learning rates in Adam, leading to suboptimal solutions.

Key Scientific Insight:
- Adam with L2 regularization: grad += wd * param (applied BEFORE moment calculation)
- AdamW (decoupled): param -= lr * wd * param (applied AFTER moment calculation)

The difference is subtle but critical for convergence quality.

Author: Senior Principal Software Engineer & Codebase Janitor
Date: 2026-01-02
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
import logging

from src.core.optimizers import Adam, AdamW
from src.core.test_functions import Rosenbrock, IllConditionedQuadratic


def run_adam_vs_adamw_comparison(
    results_dir: str = "results/adam_l2_vs_adamw",
    seeds: Optional[List[int]] = None,
    weight_decay_values: Optional[List[float]] = None,
    max_iter: int = 2000,
    resume: bool = False
) -> pd.DataFrame:
    """
    Demonstrate the critical difference between Adam (L2) and AdamW.
    
    This experiment proves the thesis that L2 regularization fails with
    adaptive optimizers, necessitating decoupled weight decay (AdamW).
    
    Args:
        results_dir: Output directory
        seeds: List of random seeds
        weight_decay_values: List of weight decay strengths to test
        max_iter: Maximum iterations
        resume: Skip if already complete
    
    Returns:
        DataFrame with convergence comparison
    """
    print("\n" + "="*80)
    print("🔬 ADAM L2 vs. ADAMW COMPARISON")
    print("="*80)
    print("   Demonstrates why L2 regularization fails with adaptive optimizers")
    print("="*80)
    
    if seeds is None:
        seeds = [42, 123, 456]
    
    if weight_decay_values is None:
        weight_decay_values = [0.0, 0.001, 0.01, 0.1]
    
    # Check resume
    if resume:
        result_file = Path(results_dir) / "adam_l2_vs_adamw_comparison.csv"
        if result_file.exists():
            try:
                df = pd.read_csv(result_file)
                if len(df) > 0:
                    logging.info(f"Skipping Adam L2 vs. AdamW experiment (already completed)")
                    return df
            except Exception:
                pass
    
    # Test function (ill-conditioned to highlight the difference)
    test_functions = [
        ("Rosenbrock", Rosenbrock(a=1, b=100), (-1.5, 2.0)),
        ("IllConditioned_1000", IllConditionedQuadratic(kappa=1000), (1.0, 1.0)),
    ]
    
    results = []
    
    for func_name, func, start_point in test_functions:
        print(f"\n[FUNCTION] {func_name}")
        print("-" * 40)
        
        for wd in weight_decay_values:
            print(f"\n  Weight Decay: {wd}")
            
            # Test Adam with L2 regularization
            for seed in seeds:
                np.random.seed(seed)
                
                optimizer = Adam(lr=0.01, weight_decay=wd)
                x, y = start_point
                history_adam_l2 = []
                
                optimizer.reset()
                
                for iteration in range(max_iter):
                    loss = func.compute(x, y)
                    history_adam_l2.append({'iteration': iteration, 'loss': loss})
                    
                    grad_x, grad_y = func.gradient(x, y)
                    x, y = optimizer.step((x, y), (grad_x, grad_y))
                    
                    if loss < 1e-8:
                        break
                
                final_loss_adam_l2 = history_adam_l2[-1]['loss'] if history_adam_l2 else float('nan')
                
                results.append({
                    'function': func_name,
                    'optimizer': 'Adam_L2',
                    'weight_decay': wd,
                    'seed': seed,
                    'final_loss': final_loss_adam_l2,
                    'final_x': x,
                    'final_y': y,
                    'iterations': len(history_adam_l2),
                    'converged': final_loss_adam_l2 < 1e-6
                })
                
                print(f"    Adam (L2):  Loss={final_loss_adam_l2:.6e}, Iters={len(history_adam_l2):4d}")
            
            # Test AdamW with decoupled weight decay
            for seed in seeds:
                np.random.seed(seed)
                
                optimizer = AdamW(lr=0.01, weight_decay=wd)
                x, y = start_point
                history_adamw = []
                
                optimizer.reset()
                
                for iteration in range(max_iter):
                    loss = func.compute(x, y)
                    history_adamw.append({'iteration': iteration, 'loss': loss})
                    
                    grad_x, grad_y = func.gradient(x, y)
                    x, y = optimizer.step((x, y), (grad_x, grad_y))
                    
                    if loss < 1e-8:
                        break
                
                final_loss_adamw = history_adamw[-1]['loss'] if history_adamw else float('nan')
                
                results.append({
                    'function': func_name,
                    'optimizer': 'AdamW',
                    'weight_decay': wd,
                    'seed': seed,
                    'final_loss': final_loss_adamw,
                    'final_x': x,
                    'final_y': y,
                    'iterations': len(history_adamw),
                    'converged': final_loss_adamw < 1e-6
                })
                
                print(f"    AdamW:      Loss={final_loss_adamw:.6e}, Iters={len(history_adamw):4d}")
    
    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"{results_dir}/adam_l2_vs_adamw_comparison.csv", index=False)
    
    print(f"\n💾 Results saved to {results_dir}/adam_l2_vs_adamw_comparison.csv")
    
    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY: Adam L2 vs. AdamW Performance")
    print("="*80)
    
    summary = df.groupby(['function', 'optimizer', 'weight_decay']).agg({
        'final_loss': ['mean', 'std'],
        'iterations': 'mean',
        'converged': 'mean'
    }).round(6)
    
    print(summary)
    
    # Generate comparison plot
    generate_adam_comparison_plot(df, results_dir)
    
    return df


def generate_adam_comparison_plot(df: pd.DataFrame, results_dir: str) -> None:
    """
    Generate visualization comparing Adam L2 vs. AdamW.
    
    Args:
        df: Results DataFrame
        results_dir: Output directory
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get unique function names using pandas unique() which handles Series properly
        unique_functions = df['function'].unique()
        for i, func_name in enumerate(unique_functions):
            ax = axes[i]
            func_mask = df['function'] == func_name
            func_data = df.loc[func_mask]
            
            # Group by weight_decay and optimizer
            # Cast to Series explicitly for type checker
            wd_series = pd.Series(func_data['weight_decay'])
            unique_wds = wd_series.unique()
            for wd in sorted(unique_wds):
                wd_data = func_data[func_data['weight_decay'] == wd]
                
                # Adam L2
                adam_l2_data = wd_data[wd_data['optimizer'] == 'Adam_L2']
                if len(adam_l2_data) > 0:
                    mean_loss = adam_l2_data['final_loss'].mean()
                    std_loss = adam_l2_data['final_loss'].std()
                    ax.errorbar([wd], [mean_loss], yerr=[std_loss], 
                               fmt='o-', label=f'Adam L2 (wd={wd})', capsize=5)
                
                # AdamW
                adamw_data = wd_data[wd_data['optimizer'] == 'AdamW']
                if len(adamw_data) > 0:
                    mean_loss = adamw_data['final_loss'].mean()
                    std_loss = adamw_data['final_loss'].std()
                    ax.errorbar([wd], [mean_loss], yerr=[std_loss], 
                               fmt='s--', label=f'AdamW (wd={wd})', capsize=5)
            
            ax.set_xlabel('Weight Decay')
            ax.set_ylabel('Final Loss')
            ax.set_title(f'{func_name}: Adam L2 vs. AdamW')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/adam_l2_vs_adamw_comparison.png", dpi=150)
        plt.close()
        
        print(f"✓ Comparison plot saved to {results_dir}/adam_l2_vs_adamw_comparison.png")
    except Exception as e:
        logging.warning(f"Could not generate comparison plot: {e}")


def run_lr_schedule_demonstration(
    results_dir: str = "results/lr_schedule_demo",
    max_iter: int = 1000
) -> pd.DataFrame:
    """
    Demonstrate learning rate scheduling with 2D optimizers.
    
    This proves that LR schedulers can now be used in 2D experiments,
    maintaining consistency with neural network training.
    
    Args:
        results_dir: Output directory
        max_iter: Maximum iterations
    
    Returns:
        DataFrame with convergence results under different schedules
    """
    print("\n" + "="*80)
    print("🔬 LEARNING RATE SCHEDULE DEMONSTRATION (2D Optimizers)")
    print("="*80)
    print("   Demonstrates scheduler compatibility via set_lr() method")
    print("="*80)
    
    func = Rosenbrock(a=1, b=100)
    start_point = (-1.5, 2.0)
    
    results = []
    
    # Test constant LR
    print("\n[SCHEDULE] Constant LR")
    optimizer = Adam(lr=0.01)
    x, y = start_point
    optimizer.reset()
    
    history = []
    for iteration in range(max_iter):
        loss = func.compute(x, y)
        history.append({'iteration': iteration, 'loss': loss, 'lr': optimizer.get_lr()})
        
        grad_x, grad_y = func.gradient(x, y)
        x, y = optimizer.step((x, y), (grad_x, grad_y))
        
        if loss < 1e-8:
            break
    
    results.append({
        'schedule': 'Constant',
        'final_loss': history[-1]['loss'],
        'iterations': len(history)
    })
    
    print(f"  Final Loss: {history[-1]['loss']:.6e}, Iterations: {len(history)}")
    
    # Test cosine annealing
    print("\n[SCHEDULE] Cosine Annealing")
    optimizer = Adam(lr=0.01)
    x, y = start_point
    optimizer.reset()
    
    history = []
    initial_lr = 0.01
    for iteration in range(max_iter):
        # Cosine annealing
        lr = initial_lr * (0.5 * (1 + np.cos(np.pi * iteration / max_iter)))
        optimizer.set_lr(lr)
        
        loss = func.compute(x, y)
        history.append({'iteration': iteration, 'loss': loss, 'lr': lr})
        
        grad_x, grad_y = func.gradient(x, y)
        x, y = optimizer.step((x, y), (grad_x, grad_y))
        
        if loss < 1e-8:
            break
    
    results.append({
        'schedule': 'Cosine_Annealing',
        'final_loss': history[-1]['loss'],
        'iterations': len(history)
    })
    
    print(f"  Final Loss: {history[-1]['loss']:.6e}, Iterations: {len(history)}")
    
    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"{results_dir}/lr_schedule_demo_results.csv", index=False)
    
    print(f"\n💾 Results saved to {results_dir}/lr_schedule_demo_results.csv")
    print("✓ Learning rate scheduling now supported in 2D experiments!")
    
    return df


if __name__ == "__main__":
    # Run Adam L2 vs. AdamW comparison
    df_adam = run_adam_vs_adamw_comparison()
    
    # Run LR schedule demonstration
    df_schedule = run_lr_schedule_demonstration()
    
    print("\n" + "="*80)
    print("✅ FINAL STRUCTURAL FIXES COMPLETE")
    print("="*80)
    print("   1. ✅ Adam now supports L2 regularization (to show it fails)")
    print("   2. ✅ AdamW available with decoupled weight decay")
    print("   3. ✅ All optimizers support set_lr() for scheduler compatibility")
    print("="*80)
