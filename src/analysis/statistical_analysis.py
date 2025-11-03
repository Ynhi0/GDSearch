"""
Statistical analysis tools for comparing optimizers.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def load_multiseed_results(pattern: str, results_dir: str = 'results') -> List[pd.DataFrame]:
    """
    Load results from multiple seed runs.
    
    Args:
        pattern: File pattern (e.g., '*AdamW*seed*.csv')
        results_dir: Results directory
        
    Returns:
        List of DataFrames
    """
    import glob
    files = glob.glob(os.path.join(results_dir, pattern))
    return [pd.read_csv(f) for f in sorted(files)]


def extract_final_metric(dfs: List[pd.DataFrame], metric: str = 'test_accuracy') -> np.ndarray:
    """Extract final metric value from each run."""
    values = []
    for df in dfs:
        eval_df = df[df['phase'] == 'eval']
        if not eval_df.empty:
            values.append(eval_df[metric].iloc[-1])
    return np.array(values)


def compare_optimizers_ttest(
    results_A: np.ndarray, 
    results_B: np.ndarray, 
    name_A: str = "Optimizer A",
    name_B: str = "Optimizer B",
    metric: str = "test_accuracy"
) -> Dict:
    """
    Perform independent t-test between two optimizers.
    
    Args:
        results_A: Array of metric values for optimizer A
        results_B: Array of metric values for optimizer B
        name_A, name_B: Names for display
        metric: Metric name
        
    Returns:
        Dictionary with test results
    """
    # Compute statistics
    mean_A = results_A.mean()
    std_A = results_A.std()
    n_A = len(results_A)
    
    mean_B = results_B.mean()
    std_B = results_B.std()
    n_B = len(results_B)
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(results_A, results_B)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_A - 1) * std_A**2 + (n_B - 1) * std_B**2) / (n_A + n_B - 2))
    cohens_d = (mean_A - mean_B) / pooled_std if pooled_std > 0 else 0.0
    
    # Confidence intervals (95%)
    ci_A = stats.t.interval(0.95, n_A - 1, loc=mean_A, scale=stats.sem(results_A))
    ci_B = stats.t.interval(0.95, n_B - 1, loc=mean_B, scale=stats.sem(results_B))
    
    result = {
        'name_A': name_A,
        'name_B': name_B,
        'mean_A': mean_A,
        'std_A': std_A,
        'n_A': n_A,
        'ci_A': ci_A,
        'mean_B': mean_B,
        'std_B': std_B,
        'n_B': n_B,
        'ci_B': ci_B,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'metric': metric
    }
    
    return result


def print_ttest_results(result: Dict):
    """Print t-test results in readable format."""
    print(f"\n{'='*70}")
    print(f"Statistical Comparison: {result['name_A']} vs {result['name_B']}")
    print(f"Metric: {result['metric']}")
    print(f"{'='*70}")
    
    print(f"\n{result['name_A']}:")
    print(f"  Mean: {result['mean_A']:.4f}")
    print(f"  Std:  {result['std_A']:.4f}")
    print(f"  N:    {result['n_A']}")
    print(f"  95% CI: [{result['ci_A'][0]:.4f}, {result['ci_A'][1]:.4f}]")
    
    print(f"\n{result['name_B']}:")
    print(f"  Mean: {result['mean_B']:.4f}")
    print(f"  Std:  {result['std_B']:.4f}")
    print(f"  N:    {result['n_B']}")
    print(f"  95% CI: [{result['ci_B'][0]:.4f}, {result['ci_B'][1]:.4f}]")
    
    print(f"\n{'─'*70}")
    print(f"Test Statistics:")
    print(f"  t-statistic: {result['t_statistic']:.4f}")
    print(f"  p-value:     {result['p_value']:.4f}")
    print(f"  Significant: {'✅ YES' if result['significant'] else '❌ NO'} (α=0.05)")
    print(f"  Effect size (Cohen's d): {result['cohens_d']:.4f}")
    
    # Interpret effect size
    d_abs = abs(result['cohens_d'])
    if d_abs < 0.2:
        effect_str = "negligible"
    elif d_abs < 0.5:
        effect_str = "small"
    elif d_abs < 0.8:
        effect_str = "medium"
    else:
        effect_str = "large"
    print(f"  Effect size interpretation: {effect_str}")
    
    # Conclusion
    print(f"\n{'─'*70}")
    diff = result['mean_A'] - result['mean_B']
    if result['significant']:
        winner = result['name_A'] if diff > 0 else result['name_B']
        print(f"✅ CONCLUSION: {winner} is statistically significantly better")
        print(f"   (p={result['p_value']:.4f} < 0.05, effect size={effect_str})")
    else:
        print(f"❌ CONCLUSION: No statistically significant difference")
        print(f"   (p={result['p_value']:.4f} ≥ 0.05)")
    
    print(f"{'='*70}\n")


def plot_comparison_with_errorbars(
    results_A: np.ndarray,
    results_B: np.ndarray,
    name_A: str = "Optimizer A",
    name_B: str = "Optimizer B",
    metric: str = "Test Accuracy",
    save_path: str = None
):
    """
    Plot comparison with error bars.
    
    Args:
        results_A, results_B: Arrays of metric values
        name_A, name_B: Optimizer names
        metric: Metric name for y-axis
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute statistics
    mean_A, std_A = results_A.mean(), results_A.std()
    mean_B, std_B = results_B.mean(), results_B.std()
    
    # Bar plot with error bars
    x = [0, 1]
    means = [mean_A, mean_B]
    stds = [std_A, std_B]
    names = [name_A, name_B]
    
    bars = ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7, 
                  color=['#1f77b4', '#ff7f0e'])
    
    # Add individual data points
    np.random.seed(42)  # For reproducible jitter
    for i, (values, xpos) in enumerate([(results_A, 0), (results_B, 1)]):
        jitter = np.random.normal(0, 0.04, size=len(values))
        ax.scatter(xpos + jitter, values, alpha=0.6, s=50, color='black', zorder=3)
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison\n(Mean ± Std, Individual Runs Shown)', 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate means
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f'{m:.4f}±{s:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def wilcoxon_signed_rank_test(results_A: np.ndarray, results_B: np.ndarray) -> Dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Use when data is paired (same initial conditions) and may not be normally distributed.
    """
    statistic, p_value = stats.wilcoxon(results_A, results_B, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def main():
    """Example usage."""
    # Generate example data
    print("Example: Comparing AdamW vs SGD+Momentum on MNIST")
    
    # Simulate results (replace with actual data loading)
    np.random.seed(42)
    adamw_results = np.random.normal(0.975, 0.005, size=5)  # Mean 97.5%, std 0.5%
    sgdm_results = np.random.normal(0.976, 0.003, size=5)   # Mean 97.6%, std 0.3%
    
    # Perform t-test
    result = compare_optimizers_ttest(
        adamw_results, 
        sgdm_results,
        name_A="AdamW",
        name_B="SGD+Momentum",
        metric="test_accuracy"
    )
    
    print_ttest_results(result)
    
    # Plot comparison
    plot_comparison_with_errorbars(
        adamw_results,
        sgdm_results,
        name_A="AdamW",
        name_B="SGD+Momentum",
        metric="Test Accuracy (%)",
        save_path="plots/statistical_comparison_example.png"
    )


if __name__ == "__main__":
    main()
