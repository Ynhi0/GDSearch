"""
Theory-Practice Convergence Validation Experiment

This module integrates theoretical convergence rate predictions with
actual neural network training results. Required by Vietnamese research proposal:
"compare observed convergence rates with theoretical predictions"

Author: GDSearch Team
Date: December 7, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Mapping
import matplotlib.pyplot as plt
import re

# Import theory-practice comparison module (tolerant to missing module or missing symbols)
try:
    import src.analysis.theory_practice_comparison as tp_comp
    predict_theoretical_rate = getattr(tp_comp, 'predict_theoretical_rate', None)
    fit_observed_rate = getattr(tp_comp, 'fit_observed_rate', None)
    compare_rates = getattr(tp_comp, 'compare_rates', None)
    generate_comparison_report = getattr(tp_comp, 'generate_comparison_report', None)
    HAS_THEORY_MODULE = all(fn is not None for fn in (predict_theoretical_rate, fit_observed_rate, compare_rates, generate_comparison_report))
except Exception:
    HAS_THEORY_MODULE = False
    predict_theoretical_rate = None
    fit_observed_rate = None
    compare_rates = None
    generate_comparison_report = None
    print("Theory-practice comparison module not available")


def extract_optimizer_from_filename(filepath: str) -> str:
    """
    Extract optimizer name from result CSV filename.
    
    Examples:
        'NN_SimpleMLP_MNIST_Adam_lr0.001_seed42.csv' -> 'Adam'
        'MNIST_SGD_Momentum_seed123.csv' -> 'SGD_Momentum'
    """
    filename = os.path.basename(filepath)
    
    # Common optimizer patterns
    optimizers = [
        'SGD_Momentum', 'SGD', 'Adam', 'AdamW', 'AMSGrad',
        'RMSprop', 'Adagrad', 'Adadelta', 'RAdam', 'AdaBound',
        'LAMB', 'Lookahead', 'SAM'
    ]
    
    for opt in optimizers:
        if opt in filename:
            return opt
    
    return 'Unknown'


def load_training_results(
    results_dir: str,
    experiment: str = 'mnist',
    required_columns: List[str] = ['epoch', 'train_loss']
) -> Dict[str, pd.DataFrame]:
    """
    Load training results from CSV files.
    
    Args:
        results_dir: Base results directory
        experiment: Experiment subdirectory (mnist, cifar10, nlp, etc.)
        required_columns: Columns that must be present
        
    Returns:
        Dictionary mapping optimizer names to DataFrames with loss histories
    """
    experiment_dir = Path(results_dir) / experiment
    
    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        return {}
    
    csv_files = list(experiment_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {experiment_dir}")
        return {}
    
    results = {}
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            # Check for required columns
            if not all(col in df.columns for col in required_columns):
                continue
            
            # Extract optimizer name
            optimizer = extract_optimizer_from_filename(str(csv_path))
            
            # Group by seed if multiple seeds present
            if 'seed' in df.columns:
                # Take average across seeds
                grouped = df.groupby('epoch')['train_loss'].mean().reset_index()
                results[optimizer] = grouped
            else:
                results[optimizer] = df[required_columns]
                
        except Exception as e:
            print(f"Failed to load {csv_path}: {e}")
            continue
    
    return results


def run_theory_practice_validation(
    results_dir: str = 'results',
    experiments: List[str] = ['mnist', 'cifar10'],
    output_dir: str = 'results/theory_practice_validation',
    problem_type: str = 'non_convex'
) -> pd.DataFrame:
    """
    Run theory-practice convergence rate comparison on actual training results.
    
    This function addresses Gap #2 from CRITICAL_GAPS_AND_FIXES.md:
    "Theory-Practice Convergence Comparison INCOMPLETE"
    
    Args:
        results_dir: Directory containing experiment results
        experiments: List of experiments to analyze
        output_dir: Directory for saving comparison results
        problem_type: Type of optimization problem ('convex', 'strongly_convex', 
                     'PL', 'non_convex')
        
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "="*80)
    print("THEORY-PRACTICE CONVERGENCE VALIDATION")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Experiments: {experiments}")
    print(f"Problem type: {problem_type}")
    print()
    
    if not HAS_THEORY_MODULE:
        print("Theory-practice comparison module not available")
        return pd.DataFrame()
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_comparisons = []
    
    for experiment in experiments:
        print(f"\n{'='*80}")
        print(f"Analyzing {experiment.upper()} results...")
        print(f"{'='*80}")
        
        # Load training results
        training_results = load_training_results(results_dir, experiment)
        
        if not training_results:
            print(f"No valid results found for {experiment}")
            continue
        
        print(f"Found {len(training_results)} optimizer results")
        
        # Analyze each optimizer
        for optimizer_name, df in training_results.items():
            print(f"\n  Analyzing {optimizer_name}...")
            
            try:
                # Extract loss history
                if 'train_loss' in df.columns:
                    loss_history = df['train_loss'].values
                elif 'loss' in df.columns:
                    loss_history = df['loss'].values
                else:
                    print(f"     No loss column found for {optimizer_name}")
                    continue
                
                # Ensure loss history is finite
                loss_history = loss_history[np.isfinite(loss_history)]
                
                if len(loss_history) < 10:
                    print(f"     Loss history too short: {len(loss_history)} steps")
                    continue
                
                # Compare with theory
                if not callable(compare_rates):
                    raise RuntimeError("compare_rates not available; ensure theory-practice module is installed")
                comparison_raw = compare_rates(
                    observed_losses=loss_history,
                    optimizer_name=optimizer_name,
                    problem_type=problem_type
                )
                # Ensure we have a plain dict with string keys for downstream processing
                comparison: Dict[str, Any]
                try:
                    if isinstance(comparison_raw, dict):
                        # Coerce keys to str to satisfy static typing and downstream consumers
                        comparison = {str(k): v for k, v in comparison_raw.items()}
                    elif isinstance(comparison_raw, Mapping):
                        try:
                            comparison = {str(k): v for k, v in comparison_raw.items()}
                        except Exception:
                            comparison = {}
                    else:
                        # Not a mapping-like object; avoid calling dict() on arbitrary objects
                        comparison = {}
                except Exception:
                    # Defensive fallback: ensure we have a dict to mutate
                    comparison = {}

                # Add metadata
                comparison['experiment'] = experiment
                comparison['dataset'] = experiment.upper()
                comparison['n_iterations'] = len(loss_history)
                comparison['initial_loss'] = loss_history[0]
                comparison['final_loss'] = loss_history[-1]
                comparison['loss_reduction'] = loss_history[0] - loss_history[-1]
                
                all_comparisons.append(comparison)
                
                # Print summary
                print(f"     Analysis complete")
                print(f"        Theoretical rate: O(k^{comparison['theoretical_rate']:.3f})")
                print(f"        Observed rate: O(k^{comparison['observed_rate']:.3f})")
                print(f"        R²: {comparison['r_squared']:.4f}")
                
                # Generate individual plot
                try:
                    plot_theory_vs_practice(
                        loss_history=loss_history,
                        optimizer_name=optimizer_name,
                        comparison=comparison,
                        output_path=os.path.join(
                            output_dir,
                            f"{experiment}_{optimizer_name}_theory_practice.png"
                        )
                    )
                except Exception as e:
                    print(f"     Plotting failed: {e}")
                    
            except Exception as e:
                print(f"     Failed to analyze {optimizer_name}: {e}")
                continue
    
    # Create summary DataFrame
    if not all_comparisons:
        print("\nNo comparisons completed")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(all_comparisons)
    
    # Save results
    csv_path = os.path.join(output_dir, "theory_practice_comparison_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Generate summary report
    try:
        generate_summary_report(df_results, output_dir)
    except Exception as e:
        print(f"Summary report generation failed: {e}")
    
    return df_results


def plot_theory_vs_practice(
    loss_history: np.ndarray,
    optimizer_name: str,
    comparison: Dict,
    output_path: str
):
    """
    Create visualization comparing theoretical and observed convergence.
    
    Args:
        loss_history: Array of loss values
        optimizer_name: Name of optimizer
        comparison: Comparison results from compare_rates()
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    
    iterations = np.arange(1, len(loss_history) + 1)
    
    # Plot 1: Loss curve with theoretical overlay
    axes[0].plot(iterations, loss_history, 'b-', linewidth=2, label='Observed')
    
    # Theoretical prediction
    theoretical_rate = comparison['theoretical_rate']
    initial_loss = loss_history[0]
    
    # Generate theoretical curve (simplified)
    if theoretical_rate < 0:  # Exponential convergence
        theoretical_loss = initial_loss * np.exp(theoretical_rate * iterations)
    else:  # Polynomial convergence
        theoretical_loss = initial_loss / (iterations ** abs(theoretical_rate))
    
    axes[0].plot(iterations, theoretical_loss, 'r--', linewidth=2, 
                 label=f'Theoretical O(k^{theoretical_rate:.2f})')
    
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title(f'{optimizer_name} - Theory vs Practice')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Log-log plot for rate analysis
    log_iterations = np.log(iterations)
    log_loss = np.log(loss_history)
    
    axes[1].scatter(log_iterations, log_loss, alpha=0.5, s=10, label='Observed')
    
    # Fitted line
    observed_rate = comparison['observed_rate']
    fitted_line = comparison['intercept'] + observed_rate * log_iterations
    axes[1].plot(log_iterations, fitted_line, 'r-', linewidth=2,
                 label=f'Fit: slope={observed_rate:.3f}')
    
    axes[1].set_xlabel('log(Iteration)')
    axes[1].set_ylabel('log(Loss)')
    axes[1].set_title(f'Convergence Rate Analysis (R²={comparison["r_squared"]:.4f})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(df_results: pd.DataFrame, output_dir: str):
    """Generate summary report with key findings"""
    report_path = os.path.join(output_dir, "theory_practice_summary.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("THEORY-PRACTICE CONVERGENCE VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"  Total comparisons: {len(df_results)}\n")
        f.write(f"  Average R²: {df_results['r_squared'].mean():.4f}\n")
        f.write(f"  Median R²: {df_results['r_squared'].median():.4f}\n\n")
        
        f.write("By Optimizer:\n")
        for opt in df_results['optimizer'].unique():
            opt_df = df_results[df_results['optimizer'] == opt]
            f.write(f"\n  {opt}:\n")
            f.write(f"    Theoretical rate: {opt_df['theoretical_rate'].mean():.4f}\n")
            f.write(f"    Observed rate: {opt_df['observed_rate'].mean():.4f}\n")
            f.write(f"    R²: {opt_df['r_squared'].mean():.4f}\n")
            f.write(f"    Experiments: {', '.join(map(str, pd.Series(opt_df['experiment']).unique()))}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("="*80 + "\n")
        f.write("High R² (>0.9): Observed convergence matches theory well\n")
        f.write("Medium R² (0.7-0.9): Reasonable agreement with minor deviations\n")
        f.write("Low R² (<0.7): Significant deviation from theory (non-convex effects)\n")
    
    print(f"Summary report saved to {report_path}")


if __name__ == '__main__':
    # Run validation on existing results
    df = run_theory_practice_validation(
        results_dir='results',
        experiments=['mnist', 'cifar10'],
        problem_type='non_convex'
    )
    
    if not df.empty:
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"Analyzed {len(df)} optimizer-experiment combinations")
        print(f"Average R²: {df['r_squared'].mean():.4f}")
