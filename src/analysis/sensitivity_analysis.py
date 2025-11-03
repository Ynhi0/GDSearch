"""
Sensitivity Analysis Tools for GDSearch

Analyzes robustness of hyperparameters by testing small perturbations
around "best" values. A robust algorithm should maintain stable performance
in a neighborhood of good hyperparameters.

Detective Question: "Is the 'best' hyperparameter value robust, or just lucky?"
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from run_nn_experiment import train_and_evaluate, result_filename


def generate_sensitivity_grid(
    center_value: float,
    perturbation_percent: float = 10.0,
    num_points: int = 5
) -> List[float]:
    """
    Generate grid of values around a center point.
    
    Args:
        center_value: Best hyperparameter value found
        perturbation_percent: Percentage to perturb (±10% by default)
        num_points: Number of points to sample (including center)
    
    Returns:
        List of values to test
    """
    if num_points == 1:
        return [center_value]
    
    # Use log scale for learning rates (multiplicative perturbation)
    perturbation = perturbation_percent / 100.0
    if center_value > 0:
        log_center = np.log10(center_value)
        log_range = log_center * perturbation
        log_values = np.linspace(log_center - log_range, log_center + log_range, num_points)
        values = 10 ** log_values
    else:
        # For parameters like momentum (additive perturbation)
        abs_range = abs(center_value) * perturbation if center_value != 0 else perturbation
        values = np.linspace(center_value - abs_range, center_value + abs_range, num_points)
    
    return values.tolist()


def run_sensitivity_experiment(
    base_config: Dict,
    param_name: str,
    param_values: List[float],
    output_dir: str = 'results/sensitivity'
) -> pd.DataFrame:
    """
    Run sensitivity analysis for a single parameter.
    
    Args:
        base_config: Base experiment configuration
        param_name: Parameter to vary ('lr', 'momentum', 'weight_decay')
        param_values: Values to test
        output_dir: Directory to save results
    
    Returns:
        DataFrame with results for each parameter value
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, value in enumerate(param_values):
        print(f"\n{'='*60}")
        print(f"Sensitivity test {i+1}/{len(param_values)}: {param_name}={value:.6f}")
        print(f"{'='*60}")
        
        # Create config for this run
        config = base_config.copy()
        config[param_name] = value
        
        # Run experiment
        try:
            df = train_and_evaluate(config)
            
            # Extract final metrics
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                final_test_acc = eval_df['test_accuracy'].iloc[-1]
                final_test_loss = eval_df['test_loss'].iloc[-1]
                
                # Calculate generalization gap
                train_df = df[df['phase'] == 'train']
                final_epoch = int(eval_df['epoch'].iloc[-1])
                train_epoch_loss = train_df[train_df['epoch'] == final_epoch]['train_loss'].mean()
                gen_gap = final_test_loss - train_epoch_loss if not pd.isna(train_epoch_loss) else None
                
                results.append({
                    'param_name': param_name,
                    'param_value': value,
                    'test_accuracy': final_test_acc,
                    'test_loss': final_test_loss,
                    'generalization_gap': gen_gap,
                    'status': 'success'
                })
                
                # Save individual run
                fname = result_filename(config).replace('.csv', f'_sensitivity_{param_name}_{i}.csv')
                out_path = os.path.join(output_dir, fname)
                df.to_csv(out_path, index=False)
                print(f"Saved: {out_path}")
                print(f"Final test accuracy: {final_test_acc:.4f}")
            else:
                results.append({
                    'param_name': param_name,
                    'param_value': value,
                    'test_accuracy': None,
                    'test_loss': None,
                    'generalization_gap': None,
                    'status': 'failed_no_eval'
                })
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'param_name': param_name,
                'param_value': value,
                'test_accuracy': None,
                'test_loss': None,
                'generalization_gap': None,
                'status': f'error: {str(e)}'
            })
    
    results_df = pd.DataFrame(results)
    
    # Save summary
    summary_path = os.path.join(output_dir, f'sensitivity_{param_name}_summary.csv')
    results_df.to_csv(summary_path, index=False)
    print(f"\n✅ Sensitivity analysis complete: {summary_path}")
    
    return results_df


def plot_sensitivity(
    results_df: pd.DataFrame,
    param_name: str,
    center_value: float,
    save_path: Optional[str] = None
):
    """
    Visualize sensitivity analysis results.
    
    Args:
        results_df: DataFrame from run_sensitivity_experiment
        param_name: Parameter name
        center_value: Original "best" value
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Filter successful runs
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if success_df.empty:
        print("⚠️ No successful runs to plot")
        return
    
    # Plot 1: Test Accuracy
    ax1 = axes[0]
    ax1.plot(success_df['param_value'], success_df['test_accuracy'], 
             'o-', linewidth=2, markersize=8, label='Test Accuracy')
    ax1.axvline(center_value, color='red', linestyle='--', linewidth=2, 
                label=f'Original best ({center_value:.6f})')
    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title(f'Sensitivity Analysis: Test Accuracy vs {param_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add stability metric (std of accuracy)
    if len(success_df) > 1:
        acc_std = success_df['test_accuracy'].std()
        acc_range = success_df['test_accuracy'].max() - success_df['test_accuracy'].min()
        ax1.text(0.02, 0.98, f'Std: {acc_std:.4f}\nRange: {acc_range:.4f}', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Generalization Gap
    ax2 = axes[1]
    gen_gap_df = success_df.dropna(subset=['generalization_gap'])
    if not gen_gap_df.empty:
        ax2.plot(gen_gap_df['param_value'], gen_gap_df['generalization_gap'],
                'o-', linewidth=2, markersize=8, color='orange', label='Gen Gap')
        ax2.axvline(center_value, color='red', linestyle='--', linewidth=2,
                   label=f'Original best ({center_value:.6f})')
        ax2.set_xlabel(param_name, fontsize=12)
        ax2.set_ylabel('Generalization Gap', fontsize=12)
        ax2.set_title(f'Sensitivity Analysis: Gen Gap vs {param_name}', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add stability metric
        if len(gen_gap_df) > 1:
            gap_std = gen_gap_df['generalization_gap'].std()
            gap_range = gen_gap_df['generalization_gap'].max() - gen_gap_df['generalization_gap'].min()
            ax2.text(0.02, 0.98, f'Std: {gap_std:.4f}\nRange: {gap_range:.4f}',
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No generalization gap data available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sensitivity plot: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_robustness(results_df: pd.DataFrame, threshold: float = 0.01) -> Dict:
    """
    Analyze robustness of hyperparameter.
    
    Args:
        results_df: DataFrame from sensitivity analysis
        threshold: Acceptable accuracy drop (default 1%)
    
    Returns:
        Dictionary with robustness metrics
    """
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if len(success_df) < 2:
        return {'status': 'insufficient_data', 'message': 'Need at least 2 successful runs'}
    
    # Find best accuracy
    best_acc = success_df['test_accuracy'].max()
    
    # Count how many runs are within threshold of best
    within_threshold = (success_df['test_accuracy'] >= best_acc - threshold).sum()
    total_runs = len(success_df)
    robustness_ratio = within_threshold / total_runs
    
    # Calculate coefficient of variation (CV)
    acc_mean = success_df['test_accuracy'].mean()
    acc_std = success_df['test_accuracy'].std()
    cv = acc_std / acc_mean if acc_mean > 0 else float('inf')
    
    # Robustness classification
    if robustness_ratio >= 0.8 and cv < 0.01:
        classification = "HIGHLY ROBUST"
    elif robustness_ratio >= 0.6 and cv < 0.02:
        classification = "MODERATELY ROBUST"
    else:
        classification = "FRAGILE"
    
    return {
        'status': 'success',
        'best_accuracy': best_acc,
        'mean_accuracy': acc_mean,
        'std_accuracy': acc_std,
        'coefficient_of_variation': cv,
        'within_threshold_ratio': robustness_ratio,
        'classification': classification,
        'interpretation': _interpret_robustness(classification)
    }


def _interpret_robustness(classification: str) -> str:
    """Provide interpretation of robustness classification."""
    interpretations = {
        "HIGHLY ROBUST": "✅ Excellent! This hyperparameter value is stable. Small changes have minimal impact. Safe for production.",
        "MODERATELY ROBUST": "⚠️ Good but be cautious. Performance degrades noticeably with perturbations. Consider wider hyperparameter search.",
        "FRAGILE": "❌ Warning! This 'best' value may be lucky. Performance is very sensitive to small changes. High risk of poor transfer to new scenarios."
    }
    return interpretations.get(classification, "Unknown classification")


def main():
    """
    Example: Run sensitivity analysis for AdamW learning rate.
    """
    # Load best config from tuning results
    print("=" * 60)
    print("Sensitivity Analysis - AdamW Learning Rate")
    print("=" * 60)
    
    # Best config from our tuning
    base_config = {
        'model': 'SimpleMLP',
        'dataset': 'MNIST',
        'optimizer': 'AdamW',
        'lr': 0.001,  # Best from tuning
        'weight_decay': 0.0,
        'epochs': 5,  # Shorter for sensitivity test
        'batch_size': 128,
        'seed': 1
    }
    
    # Generate sensitivity grid (±10% around best)
    lr_values = generate_sensitivity_grid(
        center_value=0.001,
        perturbation_percent=10.0,
        num_points=5
    )
    
    print(f"\nTesting learning rates: {lr_values}")
    print(f"Center (best) value: {base_config['lr']}")
    
    # Run sensitivity experiment
    results_df = run_sensitivity_experiment(
        base_config=base_config,
        param_name='lr',
        param_values=lr_values,
        output_dir='results/sensitivity'
    )
    
    # Analyze robustness
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 60)
    
    robustness = analyze_robustness(results_df, threshold=0.01)
    
    if robustness['status'] == 'success':
        print(f"\n📊 Results:")
        print(f"  Best Accuracy:        {robustness['best_accuracy']:.4f}")
        print(f"  Mean Accuracy:        {robustness['mean_accuracy']:.4f}")
        print(f"  Std Accuracy:         {robustness['std_accuracy']:.4f}")
        print(f"  Coeff. of Variation:  {robustness['coefficient_of_variation']:.4f}")
        print(f"  Within ±1% ratio:     {robustness['within_threshold_ratio']:.2%}")
        print(f"\n🎯 Classification: {robustness['classification']}")
        print(f"\n💡 Interpretation:\n{robustness['interpretation']}")
    else:
        print(f"⚠️ {robustness['message']}")
    
    # Plot results
    plot_sensitivity(
        results_df=results_df,
        param_name='lr',
        center_value=base_config['lr'],
        save_path='plots/sensitivity_lr.png'
    )
    
    print("\n✅ Sensitivity analysis complete!")
    print("Check: results/sensitivity/ for detailed CSVs")
    print("Check: plots/sensitivity_lr.png for visualization")


if __name__ == '__main__':
    main()
