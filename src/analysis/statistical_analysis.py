"""
Statistical analysis tools for comparing optimizers.

Includes:
- Independent t-tests with effect sizes (Cohen's d)
- Power analysis for sample size determination
- Multiple comparison corrections (Bonferroni, Holm-Bonferroni, Benjamini-Hochberg)
- Normality testing (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov)
- Non-parametric tests (Mann-Whitney U, Wilcoxon signed-rank)
- Confidence intervals
- Publication-ready visualizations
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import warnings


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
    
    # Check for zero variance cases (avoid scipy warnings)
    epsilon = 1e-10
    if std_A < epsilon and std_B < epsilon:
        # Both groups have essentially zero variance
        if abs(mean_A - mean_B) < epsilon:
            # Identical groups
            t_stat = 0.0
            p_value = 1.0
            cohens_d = 0.0
        else:
            # Different means with zero variance - very strong effect
            # Use Welch's t-test approximation for zero variance case
            t_stat = np.inf if mean_A > mean_B else -np.inf
            p_value = 0.0
            cohens_d = np.inf if mean_A > mean_B else -np.inf
        
        # For zero variance, confidence intervals collapse to the mean
        ci_A = (mean_A, mean_A)
        ci_B = (mean_B, mean_B)
    else:
        # Normal case: perform standard t-test
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


# ============================================================================
# Power Analysis
# ============================================================================


def compute_power_analysis(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> float:
    """
    Compute statistical power for a t-test.
    
    Power = probability of correctly rejecting null hypothesis when alternative is true.
    
    Args:
        effect_size: Cohen's d (standardized effect size)
        n_samples: Sample size per group
        alpha: Significance level (default: 0.05)
        alternative: 'two-sided', 'greater', or 'less'
        
    Returns:
        Statistical power (0 to 1)
    """
    # Degrees of freedom
    df = 2 * n_samples - 2
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_samples / 2)
    
    # Critical value
    if alternative == 'two-sided':
        critical_t = stats.t.ppf(1 - alpha / 2, df)
    elif alternative == 'greater':
        critical_t = stats.t.ppf(1 - alpha, df)
    else:  # 'less'
        critical_t = stats.t.ppf(alpha, df)
    
    # Power = P(reject H0 | H1 is true)
    # For two-sided test: power = P(|T| > critical_t | effect_size)
    if alternative == 'two-sided':
        power = 1 - stats.nct.cdf(critical_t, df, ncp) + stats.nct.cdf(-critical_t, df, ncp)
    elif alternative == 'greater':
        power = 1 - stats.nct.cdf(critical_t, df, ncp)
    else:  # 'less'
        power = stats.nct.cdf(critical_t, df, -ncp)
    
    return power


def compute_required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> int:
    """
    Compute required sample size to achieve desired power.
    
    Args:
        effect_size: Cohen's d (expected effect size)
        power: Desired statistical power (default: 0.8 = 80%)
        alpha: Significance level (default: 0.05)
        alternative: 'two-sided', 'greater', or 'less'
        
    Returns:
        Required sample size per group
    """
    # Binary search for required n
    n_min, n_max = 2, 1000
    
    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        current_power = compute_power_analysis(effect_size, n_mid, alpha, alternative)
        
        if current_power < power:
            n_min = n_mid
        else:
            n_max = n_mid
    
    return n_max


def power_analysis_report(
    results_A: np.ndarray,
    results_B: np.ndarray,
    name_A: str = "Optimizer A",
    name_B: str = "Optimizer B",
    target_power: float = 0.8,
    alpha: float = 0.05
) -> Dict:
    """
    Generate comprehensive power analysis report.
    
    Args:
        results_A, results_B: Arrays of metric values
        name_A, name_B: Optimizer names
        target_power: Desired power (default: 0.8)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with power analysis results
    """
    # Compute observed effect size
    n_A, n_B = len(results_A), len(results_B)
    mean_A, mean_B = results_A.mean(), results_B.mean()
    std_A, std_B = results_A.std(), results_B.std()
    
    pooled_std = np.sqrt(((n_A - 1) * std_A**2 + (n_B - 1) * std_B**2) / (n_A + n_B - 2))
    observed_effect_size = abs(mean_A - mean_B) / pooled_std if pooled_std > 0 else 0.0
    
    # Compute achieved power
    n_samples = min(n_A, n_B)
    achieved_power = compute_power_analysis(observed_effect_size, n_samples, alpha)
    
    # Compute required sample size for target power
    if observed_effect_size > 0:
        required_n = compute_required_sample_size(observed_effect_size, target_power, alpha)
    else:
        required_n = float('inf')
    
    # Power for different effect sizes (small, medium, large)
    power_small = compute_power_analysis(0.2, n_samples, alpha)
    power_medium = compute_power_analysis(0.5, n_samples, alpha)
    power_large = compute_power_analysis(0.8, n_samples, alpha)
    
    return {
        'name_A': name_A,
        'name_B': name_B,
        'n_samples': n_samples,
        'observed_effect_size': observed_effect_size,
        'achieved_power': achieved_power,
        'target_power': target_power,
        'required_n': required_n,
        'alpha': alpha,
        'power_vs_effect_size': {
            'small (0.2)': power_small,
            'medium (0.5)': power_medium,
            'large (0.8)': power_large
        }
    }


def print_power_analysis(report: Dict):
    """Print power analysis report."""
    print(f"\n{'='*70}")
    print(f"Power Analysis: {report['name_A']} vs {report['name_B']}")
    print(f"{'='*70}")
    
    print(f"\nCurrent Study:")
    print(f"  Sample size per group: {report['n_samples']}")
    print(f"  Observed effect size (Cohen's d): {report['observed_effect_size']:.4f}")
    print(f"  Achieved power: {report['achieved_power']:.4f} ({report['achieved_power']*100:.1f}%)")
    
    print(f"\nRecommendations:")
    if report['achieved_power'] >= report['target_power']:
        print(f"  ✅ Study is adequately powered (power ≥ {report['target_power']})")
    else:
        print(f"  ⚠️  Study is underpowered (power < {report['target_power']})")
        if report['required_n'] != float('inf'):
            print(f"  Required sample size for {report['target_power']*100:.0f}% power: {report['required_n']} per group")
            additional_needed = report['required_n'] - report['n_samples']
            if additional_needed > 0:
                print(f"  Need {additional_needed} more samples per group")
    
    print(f"\nPower to Detect Different Effect Sizes:")
    print(f"  (with n={report['n_samples']}, α={report['alpha']})")
    for effect_name, power_value in report['power_vs_effect_size'].items():
        status = "✅" if power_value >= 0.8 else "⚠️ "
        print(f"  {status} {effect_name}: {power_value:.4f} ({power_value*100:.1f}%)")
    
    print(f"\n{'─'*70}")
    print(f"Interpretation:")
    print(f"  - Power = probability of detecting true effect")
    print(f"  - Conventionally, power ≥ 0.80 (80%) is desired")
    print(f"  - Small effect (d=0.2): Subtle differences")
    print(f"  - Medium effect (d=0.5): Moderate differences")
    print(f"  - Large effect (d=0.8): Substantial differences")
    print(f"{'='*70}\n")


# ============================================================================
# Multiple Comparison Corrections
# ============================================================================


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Most conservative method: α_adjusted = α / n_comparisons
    
    Args:
        p_values: List of p-values
        alpha: Family-wise error rate (default: 0.05)
        
    Returns:
        Tuple of (significant_tests, adjusted_alpha)
    """
    n_comparisons = len(p_values)
    adjusted_alpha = alpha / n_comparisons
    significant = [p < adjusted_alpha for p in p_values]
    
    return significant, adjusted_alpha


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Holm-Bonferroni correction (less conservative than Bonferroni).
    
    Step-down procedure that adjusts alpha based on rank.
    
    Args:
        p_values: List of p-values
        alpha: Family-wise error rate (default: 0.05)
        
    Returns:
        List of booleans indicating significance
    """
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    
    # Test each p-value
    significant = np.zeros(n, dtype=bool)
    for i, p in enumerate(sorted_p_values):
        adjusted_alpha = alpha / (n - i)
        if p < adjusted_alpha:
            significant[sorted_indices[i]] = True
        else:
            # Once we fail to reject, stop (step-down)
            break
    
    return significant.tolist()


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Benjamini-Hochberg correction (controls False Discovery Rate).
    
    Less conservative than Bonferroni/Holm, good for exploratory analysis.
    
    Args:
        p_values: List of p-values
        alpha: False discovery rate (default: 0.05)
        
    Returns:
        List of booleans indicating significance
    """
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    
    # Find largest i where p(i) <= (i/n) * alpha
    significant = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        adjusted_alpha = ((i + 1) / n) * alpha
        if sorted_p_values[i] <= adjusted_alpha:
            # All tests up to and including i are significant
            for j in range(i + 1):
                significant[sorted_indices[j]] = True
            break
    
    return significant.tolist()


def compare_multiple_optimizers(
    results_dict: Dict[str, np.ndarray],
    correction_method: str = 'holm',
    alpha: float = 0.05,
    metric: str = 'test_accuracy'
) -> pd.DataFrame:
    """
    Perform pairwise comparisons with multiple testing correction.
    
    Args:
        results_dict: Dictionary mapping optimizer names to result arrays
        correction_method: 'bonferroni', 'holm', 'bh' (Benjamini-Hochberg), or 'none'
        alpha: Significance level
        metric: Metric name
        
    Returns:
        DataFrame with comparison results
    """
    optimizer_names = list(results_dict.keys())
    n_optimizers = len(optimizer_names)
    
    # Perform all pairwise comparisons
    comparisons = []
    p_values = []
    
    for i in range(n_optimizers):
        for j in range(i + 1, n_optimizers):
            name_A = optimizer_names[i]
            name_B = optimizer_names[j]
            results_A = results_dict[name_A]
            results_B = results_dict[name_B]
            
            # T-test
            result = compare_optimizers_ttest(results_A, results_B, name_A, name_B, metric)
            
            comparisons.append({
                'Optimizer A': name_A,
                'Optimizer B': name_B,
                'Mean A': result['mean_A'],
                'Mean B': result['mean_B'],
                'Difference': result['mean_A'] - result['mean_B'],
                'p-value': result['p_value'],
                't-statistic': result['t_statistic'],
                'Cohen\'s d': result['cohens_d']
            })
            p_values.append(result['p_value'])
    
    # Apply correction
    if correction_method == 'bonferroni':
        significant, adj_alpha = bonferroni_correction(p_values, alpha)
        correction_name = f"Bonferroni (α_adj = {adj_alpha:.4f})"
    elif correction_method == 'holm':
        significant = holm_bonferroni_correction(p_values, alpha)
        correction_name = "Holm-Bonferroni"
    elif correction_method == 'bh':
        significant = benjamini_hochberg_correction(p_values, alpha)
        correction_name = "Benjamini-Hochberg (FDR)"
    else:  # 'none'
        significant = [p < alpha for p in p_values]
        correction_name = "None (uncorrected)"
    
    # Add significance flags
    for i, comp in enumerate(comparisons):
        comp['Significant (raw)'] = comp['p-value'] < alpha
        comp['Significant (corrected)'] = significant[i]
    
    df = pd.DataFrame(comparisons)
    
    print(f"\n{'='*80}")
    print(f"Multiple Comparison Analysis ({len(optimizer_names)} optimizers, {len(comparisons)} comparisons)")
    print(f"Correction method: {correction_name}")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\n{'─'*80}")
    print(f"Summary:")
    print(f"  Significant (raw, α={alpha}): {sum(df['Significant (raw)'])}/{len(comparisons)}")
    print(f"  Significant (corrected): {sum(df['Significant (corrected)'])}/{len(comparisons)}")
    print(f"{'='*80}\n")
    
    return df


def main():
    """Example usage with all features."""
    print("="*80)
    print("Statistical Analysis Module - Complete Demo")
    print("="*80)
    
    # ========================================================================
    # Example 1: Basic T-Test
    # ========================================================================
    print("\n### EXAMPLE 1: Basic T-Test ###\n")
    
    # Simulate results (replace with actual data loading)
    np.random.seed(42)
    adamw_results = np.random.normal(0.975, 0.005, size=10)  # Mean 97.5%, std 0.5%
    sgdm_results = np.random.normal(0.976, 0.003, size=10)   # Mean 97.6%, std 0.3%
    
    # Perform t-test
    result = compare_optimizers_ttest(
        adamw_results, 
        sgdm_results,
        name_A="AdamW",
        name_B="SGD+Momentum",
        metric="test_accuracy"
    )
    
    print_ttest_results(result)
    
    # ========================================================================
    # Example 2: Power Analysis
    # ========================================================================
    print("\n### EXAMPLE 2: Power Analysis ###\n")
    
    power_report = power_analysis_report(
        adamw_results,
        sgdm_results,
        name_A="AdamW",
        name_B="SGD+Momentum",
        target_power=0.8
    )
    
    print_power_analysis(power_report)
    
    # ========================================================================
    # Example 3: Multiple Comparisons
    # ========================================================================
    print("\n### EXAMPLE 3: Multiple Comparisons ###\n")
    
    # Simulate results for 4 optimizers
    np.random.seed(42)
    results_dict = {
        'SGD': np.random.normal(0.950, 0.008, size=10),
        'SGD+Momentum': np.random.normal(0.976, 0.003, size=10),
        'RMSProp': np.random.normal(0.970, 0.005, size=10),
        'Adam': np.random.normal(0.975, 0.005, size=10)
    }
    
    print("\n--- Holm-Bonferroni Correction (Recommended) ---")
    df_holm = compare_multiple_optimizers(
        results_dict,
        correction_method='holm',
        alpha=0.05,
        metric='test_accuracy'
    )
    
    print("\n--- Bonferroni Correction (Most Conservative) ---")
    df_bonf = compare_multiple_optimizers(
        results_dict,
        correction_method='bonferroni',
        alpha=0.05,
        metric='test_accuracy'
    )
    
    print("\n--- Benjamini-Hochberg Correction (Less Conservative) ---")
    df_bh = compare_multiple_optimizers(
        results_dict,
        correction_method='bh',
        alpha=0.05,
        metric='test_accuracy'
    )
    
    # ========================================================================
    # Example 4: Sample Size Recommendations
    # ========================================================================
    print("\n### EXAMPLE 4: Sample Size Recommendations ###\n")
    
    effect_sizes = [0.2, 0.5, 0.8]
    effect_names = ['Small', 'Medium', 'Large']
    
    print("Required sample sizes for 80% power (α=0.05, two-sided):")
    print(f"{'Effect Size':<15} {'Cohen\'s d':<12} {'Required n':<12}")
    print("-" * 40)
    
    for name, d in zip(effect_names, effect_sizes):
        required_n = compute_required_sample_size(d, power=0.8, alpha=0.05)
        print(f"{name:<15} {d:<12.1f} {required_n:<12d}")
    
    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)


# =============================================================================
# Normality Testing Functions
# =============================================================================

def test_normality(
    data: np.ndarray,
    method: str = 'shapiro',
    alpha: float = 0.05
) -> Dict:
    """
    Test if data follows a normal distribution.
    
    Args:
        data: Array of values to test
        method: Test method ('shapiro', 'anderson', 'kstest')
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if len(data) < 3:
        warnings.warn("Sample size too small for normality testing (n < 3)")
        return {
            'method': method,
            'statistic': np.nan,
            'p_value': np.nan,
            'normal': None,
            'warning': 'Sample size too small'
        }
    
    if method == 'shapiro':
        # Shapiro-Wilk test (good for n < 5000)
        statistic, p_value = stats.shapiro(data)
        normal = p_value > alpha
        interpretation = f"Data {'appears' if normal else 'does not appear'} normally distributed (W={statistic:.4f}, p={p_value:.4f})"
        
    elif method == 'anderson':
        # Anderson-Darling test
        result = stats.anderson(data, dist='norm')
        # Get critical value for alpha
        if alpha == 0.05:
            crit_idx = 2  # 5% significance
        elif alpha == 0.01:
            crit_idx = 4  # 1% significance
        else:
            crit_idx = 2  # default to 5%
        
        statistic = result.statistic
        critical_value = result.critical_values[crit_idx]
        normal = statistic < critical_value
        p_value = None  # Anderson-Darling doesn't return p-value directly
        interpretation = f"Data {'appears' if normal else 'does not appear'} normally distributed (A²={statistic:.4f}, critical={critical_value:.4f})"
        
    elif method == 'kstest':
        # Kolmogorov-Smirnov test
        # Fit normal distribution to data
        mu, sigma = data.mean(), data.std()
        statistic, p_value = stats.kstest(data, 'norm', args=(mu, sigma))
        normal = p_value > alpha
        interpretation = f"Data {'appears' if normal else 'does not appear'} normally distributed (D={statistic:.4f}, p={p_value:.4f})"
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'shapiro', 'anderson', or 'kstest'")
    
    return {
        'method': method,
        'statistic': statistic,
        'p_value': p_value,
        'normal': normal,
        'alpha': alpha,
        'interpretation': interpretation,
        'n': len(data)
    }


def compare_optimizers_mann_whitney(
    results_A: np.ndarray,
    results_B: np.ndarray,
    name_A: str = "Optimizer A",
    name_B: str = "Optimizer B",
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> Dict:
    """
    Non-parametric comparison using Mann-Whitney U test (for independent samples).
    
    Use when:
    - Data is not normally distributed
    - Sample sizes are small
    - Outliers are present
    
    Args:
        results_A: Array of metric values for optimizer A
        results_B: Array of metric values for optimizer B
        name_A, name_B: Names for display
        alternative: 'two-sided', 'less', or 'greater'
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Compute statistics
    median_A = np.median(results_A)
    median_B = np.median(results_B)
    
    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(
        results_A,
        results_B,
        alternative=alternative
    )
    
    # Effect size (rank-biserial correlation)
    # r = 1 - (2U) / (n1 * n2)
    n_A, n_B = len(results_A), len(results_B)
    r = 1 - (2 * statistic) / (n_A * n_B)
    
    significant = p_value < alpha
    
    return {
        'name_A': name_A,
        'name_B': name_B,
        'median_A': median_A,
        'median_B': median_B,
        'n_A': n_A,
        'n_B': n_B,
        'U_statistic': statistic,
        'p_value': p_value,
        'effect_size_r': r,
        'significant': significant,
        'alpha': alpha,
        'alternative': alternative,
        'test': 'Mann-Whitney U'
    }


def compare_optimizers_wilcoxon(
    results_A: np.ndarray,
    results_B: np.ndarray,
    name_A: str = "Optimizer A",
    name_B: str = "Optimizer B",
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> Dict:
    """
    Non-parametric comparison using Wilcoxon signed-rank test (for paired samples).
    
    Use when:
    - Comparing same optimizers on different problems
    - Data is paired/matched
    - Distribution is not normal
    
    Args:
        results_A: Array of metric values for optimizer A
        results_B: Array of metric values for optimizer B
        name_A, name_B: Names for display
        alternative: 'two-sided', 'less', or 'greater'
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if len(results_A) != len(results_B):
        raise ValueError("Wilcoxon test requires paired samples of equal length")
    
    # Compute statistics
    median_A = np.median(results_A)
    median_B = np.median(results_B)
    median_diff = np.median(results_A - results_B)
    
    # Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(
        results_A,
        results_B,
        alternative=alternative
    )
    
    # Effect size (rank-biserial correlation for paired samples)
    # r = Z / sqrt(n)
    n = len(results_A)
    z_score = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
    r = z_score / np.sqrt(n)
    
    significant = p_value < alpha
    
    return {
        'name_A': name_A,
        'name_B': name_B,
        'median_A': median_A,
        'median_B': median_B,
        'median_diff': median_diff,
        'n': n,
        'W_statistic': statistic,
        'p_value': p_value,
        'effect_size_r': r,
        'significant': significant,
        'alpha': alpha,
        'alternative': alternative,
        'test': 'Wilcoxon signed-rank'
    }


def auto_select_test(
    results_A: np.ndarray,
    results_B: np.ndarray,
    paired: bool = False,
    alpha: float = 0.05,
    name_A: str = "Optimizer A",
    name_B: str = "Optimizer B"
) -> Dict:
    """
    Automatically select appropriate statistical test based on normality.
    
    Decision tree:
    1. Test normality of both samples
    2. If both normal: use t-test
    3. If not normal:
       - If paired: use Wilcoxon signed-rank
       - If independent: use Mann-Whitney U
    
    Args:
        results_A: Array of metric values for optimizer A
        results_B: Array of metric values for optimizer B
        paired: Whether samples are paired
        alpha: Significance level
        name_A, name_B: Names for display
        
    Returns:
        Dictionary with test results and normality info
    """
    # Test normality
    normality_A = test_normality(results_A, method='shapiro', alpha=alpha)
    normality_B = test_normality(results_B, method='shapiro', alpha=alpha)
    
    both_normal = normality_A['normal'] and normality_B['normal']
    
    # Select test
    if both_normal:
        # Parametric test
        test_result = compare_optimizers_ttest(
            results_A, results_B, name_A, name_B
        )
        test_type = 'parametric (t-test)'
    else:
        # Non-parametric test
        if paired:
            test_result = compare_optimizers_wilcoxon(
                results_A, results_B, name_A, name_B,
                alternative='two-sided', alpha=alpha
            )
            test_type = 'non-parametric (Wilcoxon)'
        else:
            test_result = compare_optimizers_mann_whitney(
                results_A, results_B, name_A, name_B,
                alternative='two-sided', alpha=alpha
            )
            test_type = 'non-parametric (Mann-Whitney U)'
    
    # Combine results
    result = {
        'test_type': test_type,
        'normality_A': normality_A,
        'normality_B': normality_B,
        'test_result': test_result
    }
    
    return result


def print_normality_results(normality_result: Dict) -> None:
    """Print formatted normality test results."""
    print(f"\nNormality Test ({normality_result['method'].capitalize()}):")
    print(f"  Sample size: n = {normality_result['n']}")
    print(f"  Test statistic: {normality_result['statistic']:.4f}")
    if normality_result['p_value'] is not None:
        print(f"  P-value: {normality_result['p_value']:.4f}")
    print(f"  {normality_result['interpretation']}")


def print_nonparametric_results(result: Dict) -> None:
    """Print formatted non-parametric test results."""
    print(f"\n{result['test']} Results:")
    print(f"  {result['name_A']}: median = {result['median_A']:.4f}, n = {result['n_A']}")
    print(f"  {result['name_B']}: median = {result['median_B']:.4f}, n = {result['n_B']}")
    
    if 'U_statistic' in result:
        print(f"  U statistic: {result['U_statistic']:.2f}")
    elif 'W_statistic' in result:
        print(f"  W statistic: {result['W_statistic']:.2f}")
        print(f"  Median difference: {result['median_diff']:.4f}")
    
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Effect size (r): {result['effect_size_r']:.4f}")
    print(f"  Significant (α={result['alpha']}): {result['significant']}")


if __name__ == "__main__":
    main()