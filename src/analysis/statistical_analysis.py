"""
Statistical analysis tools for comparing optimizers.

Includes:
- Independent t-tests with effect sizes (Cohen's d)
- Power analysis for sample size determination
- Multiple comparison corrections (Bonferroni, Holm-Bonferroni, Benjamini-Hochberg)
- Normality testing (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov)
- Non-parametric tests (Mann-Whitney U, Wilcoxon signed-rank)
- Confidence intervals
- High-quality visualizations
"""

import logging
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


def extract_final_metric(dfs: List[pd.DataFrame], metric: str = 'test_accuracy', exclude_tainted: bool = True) -> np.ndarray:
    """Extract final metric value from each run.

    Args:
        dfs: List of DataFrames produced by runs
        metric: Name of the metric to extract from `eval` phase rows
        exclude_tainted: If True, skip runs (DataFrames) where any `tainted` flag is True.
    """
    values = []
    for df in dfs:
        # If requested, skip runs that are marked tainted (OOM recovery happened)
        if exclude_tainted and 'tainted' in df.columns:
            # Some exported CSVs may have a single boolean per run; if so, use any(True) to decide
            try:
                if df['tainted'].any():
                    continue
            except Exception:
                # If type mismatch or other problem, attempt element-wise check
                try:
                    if any(bool(x) for x in df['tainted'].values):
                        continue
                except Exception:
                    # If inspection fails, fall back to include the run (don't silently drop)
                    pass

        eval_df = df[df['phase'] == 'eval']
        if not eval_df.empty:
            values.append(eval_df[metric].iloc[-1])
    return np.array(values)


def compare_two_optimizers(
    results_A: np.ndarray,
    results_B: np.ndarray,
    opt1_name: str = "Optimizer A",
    opt2_name: str = "Optimizer B",
    alpha: float = 0.05,
    metric: str = "test_accuracy",
    auto_select_test: bool = True
) -> Dict:
    """
    Compare two optimizers with statistical testing.
    
    Simplified interface that wraps compare_optimizers_ttest for backward compatibility.
    
    Args:
        results_A: Array of metric values for first optimizer
        results_B: Array of metric values for second optimizer
        opt1_name: Name of first optimizer
        opt2_name: Name of second optimizer
        alpha: Significance level (default: 0.05)
        metric: Metric name for display
        auto_select_test: If True, automatically select parametric vs non-parametric test
        
    Returns:
        Dictionary with comparison results including:
        - mean_diff: Mean difference (A - B)
        - p_value: P-value from t-test
        - cohens_d: Effect size (Cohen's d)
        - is_significant: Boolean indicating statistical significance
        - Additional fields from compare_optimizers_ttest
    """
    result = compare_optimizers_ttest(results_A, results_B, opt1_name, opt2_name, metric, auto_select_test)
    
    # Add simplified fields for backward compatibility
    result['mean_diff'] = result['mean_A'] - result['mean_B']
    result['is_significant'] = result['p_value'] < alpha
    result['alpha'] = alpha
    
    return result


def compare_optimizers_ttest(
    results_A: np.ndarray, 
    results_B: np.ndarray, 
    name_A: str = "Optimizer A",
    name_B: str = "Optimizer B",
    metric: str = "test_accuracy",
    auto_select_test: bool = True
) -> Dict:
    """
    Perform independent t-test between two optimizers.
    
    AUDIT FIX: Added automatic test selection based on normality.
    If auto_select_test=True, performs Shapiro-Wilk normality test and
    falls back to Mann-Whitney U test if normality is violated.
    
    Args:
        results_A: Array of metric values for optimizer A
        results_B: Array of metric values for optimizer B
        name_A, name_B: Names for display
        metric: Metric name
        auto_select_test: If True, automatically select parametric vs non-parametric test
        
    Returns:
        Dictionary with test results (includes 'test_used' field)
    """
    # Compute statistics
    mean_A = results_A.mean()
    std_A = results_A.std()
    n_A = len(results_A)
    
    mean_B = results_B.mean()
    std_B = results_B.std()
    n_B = len(results_B)
    
    # AUDIT FIX: Automatic normality testing
    test_used = "welch_t_test"
    normality_check = {}
    
    if auto_select_test and n_A >= 3 and n_B >= 3:
        # Shapiro-Wilk test requires n >= 3
        try:
            # CRITICAL FIX: Check for zero variance before calling Shapiro
            # scipy.stats.shapiro warns when data has range zero
            range_A = results_A.max() - results_A.min()
            range_B = results_B.max() - results_B.min()
            
            if range_A == 0 or range_B == 0:
                # Constant data: treat as degenerate case
                # For zero variance, we'll use special handling in parametric path
                # (don't force Mann-Whitney - it's less appropriate for zero variance)
                p_A = 1.0 if range_A == 0 else None
                p_B = 1.0 if range_B == 0 else None
                
                if p_A is None:
                    _, p_A = stats.shapiro(results_A)
                if p_B is None:
                    _, p_B = stats.shapiro(results_B)
                    
                normality_check = {
                    'shapiro_p_A': p_A,
                    'shapiro_p_B': p_B,
                    'normal_A': True,  # Treat constant data as "normal" for test selection
                    'normal_B': True,
                    'zero_variance_A': range_A == 0,
                    'zero_variance_B': range_B == 0
                }
                # Don't change test_used - let the zero-variance handler below deal with it
                logging.info(
                    f"Zero variance detected (range_A={range_A:.4e}, range_B={range_B:.4e}). "
                    f"Using specialized zero-variance handling."
                )
            else:
                # Normal case: data has variance, proceed with Shapiro
                _, p_A = stats.shapiro(results_A)
                _, p_B = stats.shapiro(results_B)
                normality_check = {
                    'shapiro_p_A': p_A,
                    'shapiro_p_B': p_B,
                    'normal_A': p_A > 0.05,
                    'normal_B': p_B > 0.05
                }
            
            # If either distribution fails normality, use non-parametric test
            if p_A <= 0.05 or p_B <= 0.05:
                test_used = "mann_whitney_u"
                logging.info(
                    f"Normality violated (p_A={p_A:.4f}, p_B={p_B:.4f}). "
                    f"Using Mann-Whitney U test instead of t-test."
                )
        except Exception as e:
            logging.warning(f"Normality test failed: {e}. Defaulting to Welch's t-test.")
    
    # AUDIT FIX: Branch to non-parametric test if normality violated
    if test_used == "mann_whitney_u":
        # Use Mann-Whitney U test (non-parametric alternative)
        try:
            u_stat, p_value = stats.mannwhitneyu(results_A, results_B, alternative='two-sided')
            
            # Effect size for Mann-Whitney: rank-biserial correlation
            # r = 1 - (2*U) / (n_A * n_B)
            rank_biserial = 1 - (2 * u_stat) / (n_A * n_B)
            
            # For Mann-Whitney, we don't have parametric CIs; use bootstrap or omit
            if n_A >= 2:
                ci_A = stats.t.interval(0.95, n_A - 1, loc=mean_A, scale=stats.sem(results_A))
            else:
                ci_A = (mean_A, mean_A)
            if n_B >= 2:
                ci_B = stats.t.interval(0.95, n_B - 1, loc=mean_B, scale=stats.sem(results_B))
            else:
                ci_B = (mean_B, mean_B)
            
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
                'u_statistic': u_stat,  # Mann-Whitney U statistic
                't_statistic': None,  # Not applicable for non-parametric test
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': rank_biserial,  # Rank-biserial correlation (proper name)
                'effect_size_type': 'rank_biserial',
                'cohens_d': None,  # Not applicable for non-parametric test
                'metric': metric,
                'test_used': test_used,
                'normality_check': normality_check
            }
            return result
        except Exception as e:
            logging.warning(f"Mann-Whitney U test failed: {e}. Falling back to Welch's t-test.")
            test_used = "welch_t_test"
    
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
        # AUDIT FIX: Use Welch's t-test (equal_var=False) for robustness
        # Welch's test does not assume equal variances and is more robust
        t_stat, p_value = stats.ttest_ind(results_A, results_B, equal_var=False)
        
        # Effect size: Use pooled Cohen's d (standard approach)
        # Pooled std = sqrt(((n_A-1)*std_A^2 + (n_B-1)*std_B^2) / (n_A + n_B - 2))
        # This is the standard Cohen's d formula for independent groups
        if std_A > 0 and std_B > 0:
            # Pooled standard deviation (standard Cohen's d approach)
            pooled_std = np.sqrt(((n_A - 1) * std_A**2 + (n_B - 1) * std_B**2) / (n_A + n_B - 2))
            cohens_d = (mean_A - mean_B) / pooled_std
        elif std_A > 0:
            # Only A has variance, use Glass's delta (mean_diff / std_control)
            cohens_d = (mean_A - mean_B) / std_A
        elif std_B > 0:
            # Only B has variance, use Glass's delta
            cohens_d = (mean_A - mean_B) / std_B
        else:
            cohens_d = 0.0
        
        # Confidence intervals (95%)
        # CRITICAL FIX: Check n >= 2 before computing CI to avoid invalid degrees of freedom
        if n_A >= 2:
            ci_A = stats.t.interval(0.95, n_A - 1, loc=mean_A, scale=stats.sem(results_A))
        else:
            # Cannot compute CI with n < 2, set to mean
            ci_A = (mean_A, mean_A)
        
        if n_B >= 2:
            ci_B = stats.t.interval(0.95, n_B - 1, loc=mean_B, scale=stats.sem(results_B))
        else:
            # Cannot compute CI with n < 2, set to mean
            ci_B = (mean_B, mean_B)
    
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
        'u_statistic': None,  # Not applicable for parametric test
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': cohens_d,  # Alias for consistency with non-parametric
        'effect_size_type': 'cohens_d',
        'metric': metric,
        'test_used': test_used,  # AUDIT FIX: Report which test was used
        'normality_check': normality_check  # AUDIT FIX: Report normality test results
    }
    
    return result


def print_ttest_results(result: Dict):
    """Print t-test results in readable format."""
    logging.info(f"\n{'='*70}")
    logging.info(f"Statistical Comparison: {result['name_A']} vs {result['name_B']}")
    logging.info(f"Metric: {result['metric']}")
    logging.info(f"{'='*70}")
    
    logging.info(f"\n{result['name_A']}:")
    logging.info(f"  Mean: {result['mean_A']:.4f}")
    logging.info(f"  Std:  {result['std_A']:.4f}")
    logging.info(f"  N:    {result['n_A']}")
    logging.info(f"  95% CI: [{result['ci_A'][0]:.4f}, {result['ci_A'][1]:.4f}]")
    
    logging.info(f"\n{result['name_B']}:")
    logging.info(f"  Mean: {result['mean_B']:.4f}")
    logging.info(f"  Std:  {result['std_B']:.4f}")
    logging.info(f"  N:    {result['n_B']}")
    logging.info(f"  95% CI: [{result['ci_B'][0]:.4f}, {result['ci_B'][1]:.4f}]")
    
    logging.info(f"\n{'─'*70}")
    logging.info(f"Test Statistics:")
    # AUDIT FIX: Report which test was used and appropriate statistics
    if 'test_used' in result:
        logging.info(f"  Test used: {result['test_used']}")
        if 'normality_check' in result and result['normality_check']:
            nc = result['normality_check']
            logging.info(f"  Normality (Shapiro-Wilk): A p={nc.get('shapiro_p_A', 'N/A'):.4f}, B p={nc.get('shapiro_p_B', 'N/A'):.4f}")
    
    # Display appropriate test statistic based on test type
    if result.get('t_statistic') is not None:
        logging.info(f"  t-statistic: {result['t_statistic']:.4f}")
    if result.get('u_statistic') is not None:
        logging.info(f"  U-statistic: {result['u_statistic']:.4f}")
    
    logging.info(f"  p-value:     {result['p_value']:.4f}")
    logging.info(f"  Significant: {'YES' if result['significant'] else 'NO'} (α=0.05)")
    
    # Display effect size with appropriate label
    effect_size_val = result.get('effect_size', result.get('cohens_d', 0.0))
    effect_size_type = result.get('effect_size_type', 'cohens_d')
    if effect_size_val is not None:
        logging.info(f"  Effect size ({effect_size_type}): {effect_size_val:.4f}")
        
        # Interpret effect size (same thresholds for Cohen's d and rank-biserial)
        d_abs = abs(effect_size_val)
        if d_abs < 0.2:
            effect_str = "negligible"
        elif d_abs < 0.5:
            effect_str = "small"
        elif d_abs < 0.8:
            effect_str = "medium"
        else:
            effect_str = "large"
        logging.info(f"  Effect size interpretation: {effect_str}")
    else:
        effect_str = "N/A"
    
    # Conclusion
    logging.info(f"\n{'─'*70}")
    diff = result['mean_A'] - result['mean_B']
    if result['significant']:
        winner = result['name_A'] if diff > 0 else result['name_B']
        logging.info(f"CONCLUSION: {winner} is statistically significantly better")
        logging.info(f"   (p={result['p_value']:.4f} < 0.05, effect size={effect_str})")
    else:
        logging.info(f"CONCLUSION: No statistically significant difference")
        logging.info(f"   (p={result['p_value']:.4f} ≥ 0.05)")
    
    logging.info(f"{'='*70}\n")


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
        logging.info(f"Plot saved to: {save_path}")
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
    logging.info(f"\n{'='*70}")
    logging.info(f"Power Analysis: {report['name_A']} vs {report['name_B']}")
    logging.info(f"{'='*70}")
    
    logging.info(f"\nCurrent Study:")
    logging.info(f"  Sample size per group: {report['n_samples']}")
    logging.info(f"  Observed effect size (Cohen's d): {report['observed_effect_size']:.4f}")
    logging.info(f"  Achieved power: {report['achieved_power']:.4f} ({report['achieved_power']*100:.1f}%)")
    
    logging.info(f"\nRecommendations:")
    if report['achieved_power'] >= report['target_power']:
        logging.info(f"  Study is adequately powered (power ≥ {report['target_power']})")
    else:
        logging.info(f"  Study is underpowered (power < {report['target_power']})")
        if report['required_n'] != float('inf'):
            logging.info(f"  Required sample size for {report['target_power']*100:.0f}% power: {report['required_n']} per group")
            additional_needed = report['required_n'] - report['n_samples']
            if additional_needed > 0:
                logging.info(f"  Need {additional_needed} more samples per group")
    
    logging.info(f"\nPower to Detect Different Effect Sizes:")
    logging.info(f"  (with n={report['n_samples']}, α={report['alpha']})")
    for effect_name, power_value in report['power_vs_effect_size'].items():
        status = "" if power_value >= 0.8 else "WARNING: "
        logging.info(f"  {status} {effect_name}: {power_value:.4f} ({power_value*100:.1f}%)")
    
    logging.info(f"\n{'─'*70}")
    logging.info(f"Interpretation:")
    logging.info(f"  - Power = probability of detecting true effect")
    logging.info(f"  - Conventionally, power ≥ 0.80 (80%) is desired")
    logging.info(f"  - Small effect (d=0.2): Subtle differences")
    logging.info(f"  - Medium effect (d=0.5): Moderate differences")
    logging.info(f"  - Large effect (d=0.8): Substantial differences")
    logging.info(f"{'='*70}\n")


# ============================================================================
# Friedman Test and Nemenyi Post-hoc 
# ============================================================================


def friedman_test(data: np.ndarray, optimizer_names: List[str] = None) -> Dict:
    """
    Friedman test for comparing multiple optimizers across multiple datasets/seeds.
    
    This is the MANDATORY omnibus test for ranking k > 2 algorithms across multiple
    datasets, as specified by Demšar (JMLR 2006).
    
    Args:
        data: 2D array of shape (n_datasets, n_optimizers)
              Each row is a dataset/seed, each column is an optimizer
        optimizer_names: List of optimizer names (optional)
        
    Returns:
        Dictionary with test results:
        - statistic: Friedman chi-squared statistic
        - p_value: P-value for the test
        - significant: Boolean indicating if p < 0.05
        - mean_ranks: Average rank for each optimizer (lower is better)
        - optimizer_names: Names of optimizers
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D array (n_datasets x n_optimizers)")
    
    n_datasets, n_optimizers = data.shape
    
    if n_datasets < 2:
        raise ValueError("Need at least 2 datasets/seeds for Friedman test")
    if n_optimizers < 2:
        raise ValueError("Need at least 2 optimizers for Friedman test")
    
    # Perform Friedman test (non-parametric alternative to repeated-measures ANOVA)
    statistic, p_value = stats.friedmanchisquare(*[data[:, i] for i in range(n_optimizers)])
    
    # Compute average ranks (rank within each dataset, then average across datasets)
    # Lower rank = better performance
    ranks = np.zeros_like(data)
    for i in range(n_datasets):
        # Rank in descending order (higher values = lower ranks = better)
        ranks[i, :] = stats.rankdata(-data[i, :])
    
    mean_ranks = ranks.mean(axis=0)
    
    if optimizer_names is None:
        optimizer_names = [f"Optimizer {i+1}" for i in range(n_optimizers)]
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_ranks': mean_ranks,
        'optimizer_names': optimizer_names,
        'n_datasets': n_datasets,
        'n_optimizers': n_optimizers
    }


def nemenyi_test(data: np.ndarray, optimizer_names: List[str] = None, alpha: float = 0.05) -> Dict:
    """
    Nemenyi post-hoc test for pairwise comparisons after Friedman test.
    
    This is the standard post-hoc test for ranking algorithms, as specified by
    Demšar (JMLR 2006). It controls the Family-Wise Error Rate (FWER).
    
    Args:
        data: 2D array of shape (n_datasets, n_optimizers)
        optimizer_names: List of optimizer names (optional)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with:
        - critical_distance: Critical difference for significance
        - pairwise_differences: Matrix of rank differences
        - significant_pairs: List of (i, j, rank_diff, is_significant) tuples
        - mean_ranks: Average ranks
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D array (n_datasets x n_optimizers)")
    
    n_datasets, n_optimizers = data.shape
    
    # Compute ranks
    ranks = np.zeros_like(data)
    for i in range(n_datasets):
        ranks[i, :] = stats.rankdata(-data[i, :])
    
    mean_ranks = ranks.mean(axis=0)
    
    # Critical difference for Nemenyi test
    # CD = q_α * sqrt(k(k+1) / (6N))
    # where q_α is the critical value from Studentized range distribution
    q_alpha = {
        0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
        0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
               7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}
    }
    
    # Get q value (use linear interpolation if needed)
    if alpha in q_alpha and n_optimizers in q_alpha[alpha]:
        q = q_alpha[alpha][n_optimizers]
    else:
        # Fallback: use approximate formula
        # For α=0.05: q ≈ 1.96 + 0.37 * sqrt(k-2) (rough approximation)
        q = 1.96 + 0.37 * np.sqrt(max(0, n_optimizers - 2))
    
    critical_distance = q * np.sqrt(n_optimizers * (n_optimizers + 1) / (6 * n_datasets))
    
    # Pairwise comparisons
    pairwise_diffs = np.abs(mean_ranks[:, None] - mean_ranks[None, :])
    
    significant_pairs = []
    for i in range(n_optimizers):
        for j in range(i + 1, n_optimizers):
            rank_diff = abs(mean_ranks[i] - mean_ranks[j])
            is_significant = rank_diff > critical_distance
            significant_pairs.append((i, j, rank_diff, is_significant))
    
    if optimizer_names is None:
        optimizer_names = [f"Optimizer {i+1}" for i in range(n_optimizers)]
    
    return {
        'critical_distance': critical_distance,
        'pairwise_differences': pairwise_diffs,
        'significant_pairs': significant_pairs,
        'mean_ranks': mean_ranks,
        'optimizer_names': optimizer_names,
        'alpha': alpha
    }


def print_friedman_results(results: Dict):
    """Print Friedman test results."""
    logging.info(f"\n{'='*70}")
    logging.info(f"Friedman Test Results (Omnibus Test)")
    logging.info(f"{'='*70}")
    logging.info(f"Number of datasets/seeds: {results['n_datasets']}")
    logging.info(f"Number of optimizers: {results['n_optimizers']}")
    logging.info(f"Friedman χ² statistic: {results['statistic']:.4f}")
    logging.info(f"P-value: {results['p_value']:.6f}")
    
    if results['significant']:
        logging.info(f"SIGNIFICANT: Optimizers differ significantly (p < 0.05)")
    else:
        logging.info(f"NOT SIGNIFICANT: No significant difference between optimizers")
    
    logging.info(f"\nAverage Ranks (lower is better):")
    sorted_indices = np.argsort(results['mean_ranks'])
    for rank_order, idx in enumerate(sorted_indices, 1):
        opt_name = results['optimizer_names'][idx]
        mean_rank = results['mean_ranks'][idx]
        logging.info(f"  {rank_order}. {opt_name}: {mean_rank:.2f}")
    logging.info(f"{'='*70}\n")


def print_nemenyi_results(results: Dict):
    """Print Nemenyi post-hoc test results."""
    logging.info(f"\n{'='*70}")
    logging.info(f"Nemenyi Post-hoc Test Results")
    logging.info(f"{'='*70}")
    logging.info(f"Critical Distance (CD): {results['critical_distance']:.4f}")
    logging.info(f"Significance level: α = {results['alpha']}")
    logging.info(f"\nPairwise Comparisons:")
    logging.info(f"{'─'*70}")
    
    optimizer_names = results['optimizer_names']
    for i, j, rank_diff, is_sig in results['significant_pairs']:
        status = "SIGNIFICANT" if is_sig else "  Not significant"
        logging.info(f"{status}: {optimizer_names[i]} vs {optimizer_names[j]}")
        logging.info(f"           Rank difference: {rank_diff:.4f} (CD = {results['critical_distance']:.4f})")
    
    logging.info(f"{'='*70}\n")


def plot_critical_difference_diagram(mean_ranks: np.ndarray, 
                                     optimizer_names: List[str],
                                     critical_distance: float,
                                     title: str = "Critical Difference Diagram",
                                     save_path: str = None):
    """
    Plot Critical Difference (CD) diagram for visualizing Nemenyi results.
    
    This is the standard visualization for optimizer rankings, as used in
    Demšar (JMLR 2006) and countless ML papers.
    
    Args:
        mean_ranks: Average ranks for each optimizer
        optimizer_names: List of optimizer names
        critical_distance: Critical distance from Nemenyi test
        title: Plot title
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    n_optimizers = len(mean_ranks)
    sorted_indices = np.argsort(mean_ranks)
    sorted_ranks = mean_ranks[sorted_indices]
    sorted_names = [optimizer_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot horizontal line for ranks
    ax.plot([1, n_optimizers], [0, 0], 'k-', linewidth=2)
    
    # Plot each optimizer
    y_offset = 0
    for i, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        # Alternate y positions for readability
        y_pos = y_offset + 0.2 * (i % 2)
        
        # Plot point
        ax.plot(rank, y_pos, 'o', markersize=12, color=f'C{i}')
        
        # Add label
        ax.text(rank, y_pos + 0.15, name, ha='center', va='bottom', fontsize=10)
    
    # Draw critical difference bars
    for i in range(n_optimizers):
        for j in range(i + 1, n_optimizers):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= critical_distance:
                # Not significantly different - draw connecting bar
                y_bar = -0.3 - 0.1 * (i + j) % 3
                ax.plot([sorted_ranks[i], sorted_ranks[j]], [y_bar, y_bar], 
                       'k-', linewidth=3)
    
    # Add CD annotation
    ax.annotate('', xy=(1, -0.6), xytext=(1 + critical_distance, -0.6),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax.text(1 + critical_distance/2, -0.75, f'CD = {critical_distance:.2f}',
            ha='center', fontsize=12, color='red', weight='bold')
    
    ax.set_xlim(0.5, n_optimizers + 0.5)
    ax.set_ylim(-1, 0.8)
    ax.set_xlabel('Average Rank', fontsize=14)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Critical Difference diagram saved to {save_path}")
    
    plt.show()
    return fig


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
            
            # Use effect_size field which works for both parametric and non-parametric
            effect_size_val = result.get('effect_size', result.get('cohens_d', 0.0))
            if effect_size_val is None:
                effect_size_val = 0.0
            
            comparisons.append({
                'Optimizer A': name_A,
                'Optimizer B': name_B,
                'Mean A': result['mean_A'],
                'Mean B': result['mean_B'],
                'Difference': result['mean_A'] - result['mean_B'],
                'p-value': result['p_value'],
                't-statistic': result.get('t_statistic'),
                'u-statistic': result.get('u_statistic'),
                'Effect Size': effect_size_val,
                'Effect Size Type': result.get('effect_size_type', 'unknown'),
                'Test Used': result.get('test_used', 'unknown'),
                'Cohen\'s d': result.get('cohens_d')  # Keep for backward compatibility
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
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Multiple Comparison Analysis ({len(optimizer_names)} optimizers, {len(comparisons)} comparisons)")
    logging.info(f"Correction method: {correction_name}")
    logging.info(f"{'='*80}\n")
    print(df.to_string(index=False))
    logging.info(f"\n{'─'*80}")
    logging.info(f"Summary:")
    logging.info(f"  Significant (raw, α={alpha}): {sum(df['Significant (raw)'])}/{len(comparisons)}")
    logging.info(f"  Significant (corrected): {sum(df['Significant (corrected)'])}/{len(comparisons)}")
    logging.info(f"{'='*80}\n")
    
    return df


def main():
    """Example usage with all features."""
    print("="*80)
    logging.info("Statistical Analysis Module - Complete Demo")
    print("="*80)
    
    # ========================================================================
    # Example 1: Basic T-Test
    # ========================================================================
    logging.info("\n### EXAMPLE 1: Basic T-Test ###\n")
    
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
    logging.info("\n### EXAMPLE 2: Power Analysis ###\n")
    
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
    logging.info("\n### EXAMPLE 3: Multiple Comparisons ###\n")
    
    # Simulate results for 4 optimizers
    np.random.seed(42)
    results_dict = {
        'SGD': np.random.normal(0.950, 0.008, size=10),
        'SGD+Momentum': np.random.normal(0.976, 0.003, size=10),
        'RMSProp': np.random.normal(0.970, 0.005, size=10),
        'Adam': np.random.normal(0.975, 0.005, size=10)
    }
    
    logging.info("\n--- Holm-Bonferroni Correction (Recommended) ---")
    df_holm = compare_multiple_optimizers(
        results_dict,
        correction_method='holm',
        alpha=0.05,
        metric='test_accuracy'
    )
    
    logging.info("\n--- Bonferroni Correction (Most Conservative) ---")
    df_bonf = compare_multiple_optimizers(
        results_dict,
        correction_method='bonferroni',
        alpha=0.05,
        metric='test_accuracy'
    )
    
    logging.info("\n--- Benjamini-Hochberg Correction (Less Conservative) ---")
    df_bh = compare_multiple_optimizers(
        results_dict,
        correction_method='bh',
        alpha=0.05,
        metric='test_accuracy'
    )
    
    # ========================================================================
    # Example 4: Sample Size Recommendations
    # ========================================================================
    logging.info("\n### EXAMPLE 4: Sample Size Recommendations ###\n")
    
    effect_sizes = [0.2, 0.5, 0.8]
    effect_names = ['Small', 'Medium', 'Large']
    
    logging.info("Required sample sizes for 80% power (α=0.05, two-sided):")
    cohens_d_label = "Cohen's d"
    logging.info(f"{'Effect Size':<15} {cohens_d_label:<12} {'Required n':<12}")
    print("-" * 40)
    
    for name, d in zip(effect_names, effect_sizes):
        required_n = compute_required_sample_size(d, power=0.8, alpha=0.05)
        logging.info(f"{name:<15} {d:<12.1f} {required_n:<12d}")
    
    print("\n" + "="*80)
    logging.info("Demo complete!")
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
        # CRITICAL FIX: Check for zero variance before calling Shapiro
        data_range = data.max() - data.min()
        if data_range == 0:
            # Constant data: treat as non-normal (degenerate case)
            return {
                'method': 'shapiro',
                'statistic': 1.0,
                'p_value': 1.0,
                'normal': False,
                'zero_variance': True,
                'interpretation': f"Data has zero variance (constant values). Treated as non-normal."
            }
        
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
    logging.info(f"\nNormality Test ({normality_result['method'].capitalize()}):")
    logging.info(f"  Sample size: n = {normality_result['n']}")
    logging.info(f"  Test statistic: {normality_result['statistic']:.4f}")
    if normality_result['p_value'] is not None:
        logging.info(f"  P-value: {normality_result['p_value']:.4f}")
    logging.info(f"  {normality_result['interpretation']}")


def print_nonparametric_results(result: Dict) -> None:
    """Print formatted non-parametric test results."""
    logging.info(f"\n{result['test']} Results:")
    logging.info(f"  {result['name_A']}: median = {result['median_A']:.4f}, n = {result['n_A']}")
    logging.info(f"  {result['name_B']}: median = {result['median_B']:.4f}, n = {result['n_B']}")
    
    if 'U_statistic' in result:
        logging.info(f"  U statistic: {result['U_statistic']:.2f}")
    elif 'W_statistic' in result:
        logging.info(f"  W statistic: {result['W_statistic']:.2f}")
        logging.info(f"  Median difference: {result['median_diff']:.4f}")
    
    logging.info(f"  P-value: {result['p_value']:.4f}")
    logging.info(f"  Effect size (r): {result['effect_size_r']:.4f}")
    logging.info(f"  Significant (α={result['alpha']}): {result['significant']}")


if __name__ == "__main__":
    main()