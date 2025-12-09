#!/usr/bin/env python3
"""
Optimizer Comparison Matrix

Performs all-vs-all pairwise comparisons of optimizers with:
1. Statistical significance testing (t-tests)
2. Effect size computation (Cohen's d)
3. Win/loss/tie matrix
4. Heatmap visualization
5. Comprehensive summary report
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.statistical_analysis import compare_two_optimizers


def load_optimizer_results(
    results_dir: str,
    optimizers: List[str],
    metric: str = 'test_accuracy'
) -> Dict[str, np.ndarray]:
    """
    Load results for all optimizers.
    
    Args:
        results_dir: Directory containing result CSVs
        optimizers: List of optimizer names
        metric: Metric to extract
        
    Returns:
        Dictionary mapping optimizer names to metric arrays
    """
    optimizer_results = {}
    results_path = Path(results_dir)
    
    for optimizer in optimizers:
        # Find all CSV files for this optimizer
        pattern = f"*{optimizer}*seed*.csv"
        files = list(results_path.glob(pattern))
        
        if not files:
            print(f"No results found for {optimizer}")
            continue
        
        metrics = []
        for file in files:
            try:
                df = pd.read_csv(file)
                eval_df = df[df['phase'] == 'eval']
                if not eval_df.empty:
                    final_value = eval_df[metric].iloc[-1]
                    metrics.append(final_value)
            except Exception as e:
                print(f"  Error reading {file.name}: {e}")
        
        if metrics:
            optimizer_results[optimizer] = np.array(metrics)
            print(f"{optimizer}: {len(metrics)} runs loaded")
    
    return optimizer_results


def create_comparison_matrix(
    optimizer_results: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create all-vs-all comparison matrices.
    
    Args:
        optimizer_results: Dictionary mapping optimizer names to metric arrays
        alpha: Significance level
        
    Returns:
        Tuple of (p_value_matrix, effect_size_matrix, win_loss_matrix)
    """
    optimizers = list(optimizer_results.keys())
    n = len(optimizers)
    
    # Initialize matrices
    p_values = np.ones((n, n))
    effect_sizes = np.zeros((n, n))
    win_loss = np.zeros((n, n))  # +1 for win, 0 for tie, -1 for loss
    
    # Perform all pairwise comparisons
    for i, opt_a in enumerate(optimizers):
        for j, opt_b in enumerate(optimizers):
            if i == j:
                continue
            
            results_a = optimizer_results[opt_a]
            results_b = optimizer_results[opt_b]
            
            # Perform statistical comparison
            comparison = compare_two_optimizers(
                results_a, results_b,
                opt1_name=opt_a,
                opt2_name=opt_b,
                alpha=alpha
            )
            
            p_values[i, j] = comparison['p_value']
            effect_sizes[i, j] = comparison['cohens_d']
            
            # Determine win/loss/tie
            if comparison['is_significant']:
                if comparison['mean_diff'] > 0:
                    win_loss[i, j] = 1  # opt_a wins
                else:
                    win_loss[i, j] = -1  # opt_b wins
            # else: tie (remains 0)
    
    # Convert to DataFrames
    p_value_df = pd.DataFrame(p_values, index=optimizers, columns=optimizers)
    effect_size_df = pd.DataFrame(effect_sizes, index=optimizers, columns=optimizers)
    win_loss_df = pd.DataFrame(win_loss, index=optimizers, columns=optimizers)
    
    return p_value_df, effect_size_df, win_loss_df


def plot_comparison_heatmaps(
    p_value_df: pd.DataFrame,
    effect_size_df: pd.DataFrame,
    win_loss_df: pd.DataFrame,
    save_path: str = None
):
    """
    Plot comparison matrices as heatmaps.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: P-values
    sns.heatmap(p_value_df, annot=True, fmt='.3f', cmap='RdYlGn_r',
               vmin=0, vmax=0.1, center=0.05,
               linewidths=0.5, square=True, ax=axes[0],
               cbar_kws={'label': 'p-value'})
    axes[0].set_title('P-Values (Row vs Column)\nGreen = Significant Difference',
                     fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Optimizer B', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Optimizer A', fontsize=11, fontweight='bold')
    
    # Plot 2: Effect sizes (Cohen's d)
    sns.heatmap(effect_size_df, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, vmin=-2, vmax=2,
               linewidths=0.5, square=True, ax=axes[1],
               cbar_kws={'label': "Cohen's d"})
    axes[1].set_title("Effect Sizes (Row - Column)\nRed = Row Better, Blue = Column Better",
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Optimizer B', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Optimizer A', fontsize=11, fontweight='bold')
    
    # Plot 3: Win/Loss matrix
    sns.heatmap(win_loss_df, annot=True, fmt='.0f', cmap='RdYlGn',
               center=0, vmin=-1, vmax=1,
               linewidths=0.5, square=True, ax=axes[2],
               cbar_kws={'label': '+1=Win, 0=Tie, -1=Loss'})
    axes[2].set_title('Win/Loss Matrix (Row vs Column)\nGreen = Row Wins',
                     fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Optimizer B', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Optimizer A', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison heatmaps saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def generate_comparison_report(
    optimizer_results: Dict[str, np.ndarray],
    p_value_df: pd.DataFrame,
    effect_size_df: pd.DataFrame,
    win_loss_df: pd.DataFrame,
    save_path: str = None
) -> str:
    """
    Generate comprehensive comparison report.
    """
    report = []
    report.append("="*80)
    report.append("OPTIMIZER COMPARISON MATRIX - COMPREHENSIVE REPORT")
    report.append("="*80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-"*80)
    for optimizer, results in optimizer_results.items():
        report.append(f"{optimizer}:")
        report.append(f"  Mean: {np.mean(results):.4f}")
        report.append(f"  Std:  {np.std(results):.4f}")
        report.append(f"  N:    {len(results)}")
        report.append("")
    
    # Win/Loss summary
    report.append("WIN/LOSS SUMMARY")
    report.append("-"*80)
    optimizers = list(optimizer_results.keys())
    
    for optimizer in optimizers:
        wins = (win_loss_df.loc[optimizer] == 1).sum()
        losses = (win_loss_df.loc[optimizer] == -1).sum()
        ties = (win_loss_df.loc[optimizer] == 0).sum() - 1  # Exclude self-comparison
        
        report.append(f"{optimizer}: {wins}W - {losses}L - {ties}T")
    
    report.append("")
    
    # Ranking
    report.append("OPTIMIZER RANKING (by mean performance)")
    report.append("-"*80)
    means = {opt: np.mean(results) for opt, results in optimizer_results.items()}
    ranked = sorted(means.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (optimizer, mean) in enumerate(ranked, 1):
        wins = (win_loss_df.loc[optimizer] == 1).sum()
        report.append(f"{rank}. {optimizer:<15} - Mean: {mean:.4f}, Wins: {wins}")
    
    report.append("")
    
    # Significant pairwise comparisons
    report.append("SIGNIFICANT PAIRWISE DIFFERENCES (p < 0.05)")
    report.append("-"*80)
    
    sig_count = 0
    for i, opt_a in enumerate(optimizers):
        for j, opt_b in enumerate(optimizers):
            if i >= j:  # Only upper triangle
                continue
            
            p_val = p_value_df.loc[opt_a, opt_b]
            if p_val < 0.05:
                effect = effect_size_df.loc[opt_a, opt_b]
                mean_a = np.mean(optimizer_results[opt_a])
                mean_b = np.mean(optimizer_results[opt_b])
                
                if mean_a > mean_b:
                    winner, loser = opt_a, opt_b
                else:
                    winner, loser = opt_b, opt_a
                
                report.append(f"{winner} > {loser}: "
                            f"p={p_val:.4f}, d={abs(effect):.2f} "
                            f"(Δ={abs(mean_a - mean_b):.4f})")
                sig_count += 1
    
    if sig_count == 0:
        report.append("No significant differences found.")
    
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Comparison report saved to: {save_path}")
    
    print(report_text)
    return report_text


def run_optimizer_comparison_matrix(
    results_dir: str,
    optimizers: List[str],
    metric: str = 'test_accuracy',
    output_dir: str = 'results/optimizer_comparison',
    alpha: float = 0.05
):
    """
    Run complete optimizer comparison matrix analysis.
    
    Args:
        results_dir: Directory containing optimizer result CSVs
        optimizers: List of optimizer names to compare
        metric: Metric to compare
        output_dir: Output directory for results
        alpha: Significance level
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("OPTIMIZER COMPARISON MATRIX ANALYSIS")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Optimizers: {optimizers}")
    print(f"Metric: {metric}")
    print(f"Significance level: {alpha}")
    print("="*80)
    
    # Load results
    print("\n📂 Loading optimizer results...")
    optimizer_results = load_optimizer_results(results_dir, optimizers, metric)
    
    if len(optimizer_results) < 2:
        print("Need at least 2 optimizers with results!")
        return
    
    # Create comparison matrices
    print("\n🔬 Computing pairwise comparisons...")
    p_value_df, effect_size_df, win_loss_df = create_comparison_matrix(
        optimizer_results, alpha
    )
    
    # Save matrices
    p_value_df.to_csv(f"{output_dir}/p_values.csv")
    effect_size_df.to_csv(f"{output_dir}/effect_sizes.csv")
    win_loss_df.to_csv(f"{output_dir}/win_loss_matrix.csv")
    print(f"Matrices saved to {output_dir}/")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_comparison_heatmaps(
        p_value_df, effect_size_df, win_loss_df,
        save_path=f"{output_dir}/comparison_heatmaps.png"
    )
    
    # Generate report
    print("\n📝 Generating comprehensive report...")
    generate_comparison_report(
        optimizer_results,
        p_value_df,
        effect_size_df,
        win_loss_df,
        save_path=f"{output_dir}/comparison_report.txt"
    )
    
    print("\n" + "="*80)
    print("OPTIMIZER COMPARISON MATRIX COMPLETE!")
    print("="*80)


def main():
    """Run optimizer comparison matrix analysis."""
    
    # Configuration
    results_dir = 'results/experiments/mnist'
    optimizers = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW', 'AMSGrad']
    
    run_optimizer_comparison_matrix(
        results_dir=results_dir,
        optimizers=optimizers,
        metric='test_accuracy',
        output_dir='results/optimizer_comparison'
    )


if __name__ == '__main__':
    main()
