"""
Ablation study comparing optimizer progression: SGD → SGD+Momentum → RMSProp → Adam → AdamW → AMSGrad

Demonstrates incremental algorithmic improvements on a single challenging task (Rosenbrock).
Outputs:
- CSV with convergence metrics
- Figure showing loss curves and final performance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

from src.core.test_functions import Rosenbrock
from src.core.optimizers import SGD, SGDMomentum, RMSProp, Adam, AdamW, AMSGrad


def run_optimizer_ablation(
    test_function,
    initial_point: tuple,
    max_iterations: int = 10000,
    results_dir: str = 'results',
    plots_dir: str = 'plots'
) -> pd.DataFrame:
    """
    Run ablation study comparing optimizer variants.
    
    Args:
        test_function: Test function instance
        initial_point: Starting (x, y)
        max_iterations: Number of iterations
        results_dir: Directory for CSV output
        plots_dir: Directory for plots
        
    Returns:
        DataFrame with summary metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define optimizer sequence (progressive improvements)
    optimizers = [
        ('SGD', SGD(lr=0.001)),
        ('SGD+Momentum', SGDMomentum(lr=0.01, beta=0.9)),
        ('RMSProp', RMSProp(lr=0.01, decay_rate=0.9)),
        ('Adam', Adam(lr=0.01, beta1=0.9, beta2=0.999)),
        ('AdamW', AdamW(lr=0.01, beta1=0.9, beta2=0.999, weight_decay=0.01)),
        ('AMSGrad', AMSGrad(lr=0.01, beta1=0.9, beta2=0.999)),
    ]
    
    # Storage for trajectories
    trajectories = {}
    summary_metrics = []
    
    print(f"\n{'='*70}")
    print(f"Optimizer Ablation Study: {test_function.__class__.__name__}")
    print(f"Initial point: {initial_point}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*70}\n")
    
    for opt_name, optimizer in optimizers:
        optimizer.reset()
        x, y = initial_point
        
        history = {
            'iteration': [],
            'loss': [],
            'grad_norm': [],
            'x': [],
            'y': []
        }
        
        for i in range(max_iterations):
            try:
                loss = test_function.compute(x, y)
                grad_x, grad_y = test_function.gradient(x, y)
                
                # Overflow protection
                if not np.isfinite(loss) or not np.isfinite(grad_x) or not np.isfinite(grad_y):
                    raise OverflowError("Non-finite gradient or loss")
                
                grad_norm = np.sqrt(grad_x**2 + grad_y**2)
                
                if not np.isfinite(grad_norm):
                    raise OverflowError("Non-finite grad_norm")
                
                history['iteration'].append(i)
                history['loss'].append(loss)
                history['grad_norm'].append(grad_norm)
                history['x'].append(x)
                history['y'].append(y)
                
                x, y = optimizer.step((x, y), (grad_x, grad_y))
                
                # Check if step produced non-finite values
                if not np.isfinite(x) or not np.isfinite(y):
                    raise OverflowError("Non-finite parameters after step")
            
            except (OverflowError, FloatingPointError, RuntimeWarning):
                # Divergence detected; fill remaining with NaN
                for j in range(i, max_iterations):
                    history['iteration'].append(j)
                    history['loss'].append(np.nan)
                    history['grad_norm'].append(np.nan)
                    history['x'].append(np.nan)
                    history['y'].append(np.nan)
                break
        
        trajectories[opt_name] = history
        
        # Summary statistics
        final_loss = history['loss'][-1]
        final_grad = history['grad_norm'][-1]
        
        # Handle NaN (divergence)
        if not np.isfinite(final_loss):
            final_loss = np.inf
            final_grad = np.inf
            min_loss = np.inf
            converged_iter = None
            precise_converged_iter = None
        else:
            min_loss = min([l for l in history['loss'] if np.isfinite(l)], default=np.inf)
            
            # Find iteration where loss < 1e-3 (practical convergence)
            converged_iter = None
            for i, loss_val in enumerate(history['loss']):
                if np.isfinite(loss_val) and loss_val < 1e-3:
                    converged_iter = i
                    break
            
            # Find iteration where grad_norm < 1e-6 (precise convergence)
            precise_converged_iter = None
            for i, gn in enumerate(history['grad_norm']):
                if np.isfinite(gn) and gn < 1e-6:
                    precise_converged_iter = i
                    break
        
        summary_metrics.append({
            'Optimizer': opt_name,
            'Final Loss': final_loss,
            'Final Grad Norm': final_grad,
            'Min Loss': min_loss,
            'Iterations to Loss<1e-3': converged_iter if converged_iter else max_iterations,
            'Iterations to GradNorm<1e-6': precise_converged_iter if precise_converged_iter else max_iterations,
            'Converged (loss<1e-3)': converged_iter is not None,
            'Converged (grad<1e-6)': precise_converged_iter is not None
        })
        
        print(f"{opt_name:20s} | Final Loss: {final_loss:12.6e} | "
              f"Converged (loss<1e-3): {'YES' if converged_iter else 'NO':3s} at iter {converged_iter if converged_iter else ('DIV' if np.isinf(final_loss) else '>10k')}")
    
    # Save summary CSV
    df_summary = pd.DataFrame(summary_metrics)
    summary_path = os.path.join(results_dir, 'optimizer_ablation_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Define color map for consistent coloring
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(optimizers)))
    
    # Plot 1: Loss curves (log scale)
    ax = axes[0, 0]
    for (opt_name, _), color in zip(optimizers, colors):
        hist = trajectories[opt_name]
        # Sample every 10 iterations for clarity; filter out non-finite
        sample_iters = []
        sample_loss = []
        for i in range(0, len(hist['loss']), 10):
            if np.isfinite(hist['loss'][i]) and hist['loss'][i] > 0:
                sample_iters.append(hist['iteration'][i])
                sample_loss.append(hist['loss'][i])
        if sample_loss:
            ax.plot(sample_iters, sample_loss, label=opt_name, linewidth=2, color=color, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Loss Convergence (Ablation Study)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm (log scale)
    ax = axes[0, 1]
    for (opt_name, _), color in zip(optimizers, colors):
        hist = trajectories[opt_name]
        sample_iters = []
        sample_grad = []
        for i in range(0, len(hist['grad_norm']), 10):
            if np.isfinite(hist['grad_norm'][i]) and hist['grad_norm'][i] > 0:
                sample_iters.append(hist['iteration'][i])
                sample_grad.append(hist['grad_norm'][i])
        if sample_grad:
            ax.plot(sample_iters, sample_grad, label=opt_name, linewidth=2, color=color, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Gradient Norm Convergence', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Bar chart of final loss
    ax = axes[1, 0]
    opt_names = [m['Optimizer'] for m in summary_metrics]
    final_losses = [m['Final Loss'] for m in summary_metrics]
    # Replace inf with a large value for visualization
    final_losses_plot = [fl if np.isfinite(fl) else 1e3 for fl in final_losses]
    bars = ax.bar(range(len(opt_names)), final_losses_plot, color=colors, alpha=0.8)
    ax.set_xticks(range(len(opt_names)))
    ax.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Final Loss (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Final Loss Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate bars
    for i, (name, loss, loss_plot) in enumerate(zip(opt_names, final_losses, final_losses_plot)):
        label = f'{loss:.2e}' if np.isfinite(loss) else 'DIV'
        ax.text(i, loss_plot * 1.5, label, ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Plot 4: Convergence speed (iterations to loss < 1e-3)
    ax = axes[1, 1]
    iters_to_converge = [m['Iterations to Loss<1e-3'] for m in summary_metrics]
    bars = ax.bar(range(len(opt_names)), iters_to_converge, color=colors, alpha=0.8)
    ax.set_xticks(range(len(opt_names)))
    ax.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Iterations to Loss < 1e-3', fontsize=11)
    ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max_iterations * 1.1])
    
    # Annotate bars
    for i, (name, it) in enumerate(zip(opt_names, iters_to_converge)):
        label = f'{it}' if it < max_iterations else '>10k'
        ax.text(i, it + max_iterations * 0.02, label, ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Optimizer Ablation: {test_function.__class__.__name__}\n'
                 f'Initial point: {initial_point}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, 'optimizer_ablation_study.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_path}")
    plt.close()
    
    # Print summary table
    print(f"\n{'='*70}")
    print("Ablation Summary:")
    print(f"{'='*70}")
    print(df_summary.to_string(index=False))
    print(f"{'='*70}\n")
    
    return df_summary


def main():
    """Run ablation study on Rosenbrock function."""
    print("="*70)
    print("Optimizer Ablation Study")
    print("="*70)
    
    # Use standard challenging initial point for Rosenbrock
    initial_point = (-1.5, 2.0)
    
    # Initialize Rosenbrock function
    rosenbrock = Rosenbrock(a=1, b=100)
    
    # Run ablation
    df_summary = run_optimizer_ablation(
        test_function=rosenbrock,
        initial_point=initial_point,
        max_iterations=10000,
        results_dir='results',
        plots_dir='plots'
    )
    
    print("✅ Ablation study complete!")


if __name__ == '__main__':
    main()
