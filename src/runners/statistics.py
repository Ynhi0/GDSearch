"""
Statistical Analysis Module for GDSearch.

Handles cross-experiment statistical comparisons including:
- Paired t-tests
- Multiple comparisons correction (Benjamini-Hochberg FDR)
- Effect size calculation (Cohen's d with confidence intervals)
- Result aggregation and reporting
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None  # type: ignore[assignment]


def aggregate_experiment_results(experiment_results: Dict[str, pd.DataFrame],
                                 results_dir: Path) -> tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]], Path]:
    """
    Aggregate results across all experiments and compute summary statistics.
    
    Args:
        experiment_results: Dict mapping experiment names to result DataFrames
        results_dir: Directory to save analysis outputs
    
    Returns:
        Tuple of (aggregated_df, optimizer_performance, analysis_dir)
    """
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)

    # Collect optimizer performance across experiments
    optimizer_performance: Dict[str, List[Dict[str, Any]]] = {}

    for exp_name, df in experiment_results.items():
        if df is None or len(df) == 0:
            continue

        # Group by optimizer and compute average accuracy
        for opt in df['optimizer'].unique():
            opt_df = df[df['optimizer'] == opt]
            avg_accuracy = opt_df['test_accuracy'].mean() if 'test_accuracy' in opt_df.columns else None

            if opt not in optimizer_performance:
                optimizer_performance[opt] = []

            optimizer_performance[opt].append({
                'experiment': exp_name,
                'accuracy': avg_accuracy,
                'count': len(opt_df)
            })

    # Create aggregated summary
    agg_data = []
    for opt, performances in optimizer_performance.items():
        accuracies = [p['accuracy'] for p in performances if p['accuracy'] is not None]
        total_experiments = sum(p['count'] for p in performances)

        agg_data.append({
            'optimizer': opt,
            'avg_accuracy_across_experiments': np.mean(accuracies) if accuracies else None,
            'std_accuracy_across_experiments': np.std(accuracies, ddof=1) if len(accuracies) > 1 else None,
            'experiments_count': len(performances),
            'total_runs': total_experiments
        })

    agg_df = pd.DataFrame(agg_data)

    # Save aggregated results
    agg_path = analysis_dir / "aggregated_optimizer_performance.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"\n   Aggregated optimizer performance saved to {agg_path}")

    # Print summary
    print("\n   Cross-Experiment Summary:")
    for _, row in agg_df.iterrows():
        opt_name = row['optimizer']
        avg_val = row['avg_accuracy_across_experiments']
        is_na = pd.isna(avg_val)

        if isinstance(is_na, np.ndarray):
            is_na_bool = bool(is_na.any())
        else:
            is_na_bool = bool(is_na)

        if is_na_bool or avg_val is None:
            acc_str = "N/A"
        else:
            try:
                acc_str = f"{float(avg_val):.2f}%"
            except Exception:
                acc_str = "N/A"
        exp_count = row.get('experiments_count', 0)
        if exp_count is None:
            exp_count_int = 0
        else:
            try:
                exp_count_int = int(exp_count)
            except Exception:
                exp_count_int = 0
        print(f"      {row['optimizer']:20s}: {acc_str} (across {exp_count_int} experiments)")

    return agg_df, optimizer_performance, analysis_dir


def perform_statistical_comparison(optimizer_performance: Dict[str, List[Dict[str, Any]]],
                                   analysis_dir: Path) -> List[Dict[str, Any]]:
    """
    Perform pairwise statistical comparisons between optimizers.
    
    Includes:
    - Paired t-tests
    - Benjamini-Hochberg FDR correction
    - Cohen's d with confidence intervals
    
    Args:
        optimizer_performance: Dict mapping optimizers to their performance across experiments
        analysis_dir: Directory to save statistical results
    
    Returns:
        List of statistical comparison results
    """
    if not HAS_SCIPY or len(optimizer_performance) < 2:
        return []

    from src.analysis.statistical_analysis import safe_ttest_rel, cohens_d_ci_paired, interpret_cohens_d

    print("\n   Cross-Experiment Statistical Analysis:")

    stat_results = []
    optimizers = list(optimizer_performance.keys())
    raw_p_values = []

    # Perform pairwise comparisons
    for i, opt_a in enumerate(optimizers):
        for opt_b in optimizers[i+1:]:
            # Get comparable experiments
            exps_a = {p['experiment']: p['accuracy'] 
                     for p in optimizer_performance[opt_a] 
                     if p['accuracy'] is not None}
            exps_b = {p['experiment']: p['accuracy'] 
                     for p in optimizer_performance[opt_b] 
                     if p['accuracy'] is not None}

            common_exps = set(exps_a.keys()) & set(exps_b.keys())

            if len(common_exps) >= 2:
                vals_a = [exps_a[e] for e in common_exps]
                vals_b = [exps_b[e] for e in common_exps]

                try:
                    # Paired t-test
                    t_stat, p_scalar = safe_ttest_rel(vals_a, vals_b)

                    # Effect size with confidence interval
                    cohens_d, d_ci_lower, d_ci_upper = cohens_d_ci_paired(
                        vals_a, vals_b, confidence=0.95
                    )
                    d_interpretation = interpret_cohens_d(cohens_d)

                    stat_results.append({
                        'optimizer_a': opt_a,
                        'optimizer_b': opt_b,
                        'n_experiments': len(common_exps),
                        'mean_diff': float(np.mean(vals_a) - np.mean(vals_b)),
                        't_statistic': t_stat,
                        'p_value': p_scalar,
                        'cohens_d': cohens_d,
                        'cohens_d_ci_lower': d_ci_lower,
                        'cohens_d_ci_upper': d_ci_upper,
                        'effect_interpretation': d_interpretation,
                        'significant': False  # Updated after FDR correction
                    })
                    raw_p_values.append(p_scalar)

                except (KeyError, IndexError, ValueError, TypeError, RuntimeError) as e:
                    logging.debug("Could not compare %s vs %s: %s", opt_a, opt_b, e)

    # Apply Benjamini-Hochberg FDR correction
    if stat_results and raw_p_values:
        valid_indices = [i for i, p in enumerate(raw_p_values) if not math.isnan(p)]
        valid_p_values = [raw_p_values[i] for i in valid_indices]

        if len(valid_p_values) > 0:
            n_tests = len(valid_p_values)
            sorted_indices = sorted(range(len(valid_p_values)), 
                                   key=lambda i: valid_p_values[i])
            adjusted_p_values = [float("nan")] * len(valid_p_values)

            for rank, idx in enumerate(sorted_indices, start=1):
                adj_p = min(1.0, valid_p_values[idx] * n_tests / rank)
                adjusted_p_values[idx] = adj_p

            # Update significance based on adjusted p-values
            for i, valid_idx in enumerate(valid_indices):
                stat_results[valid_idx]['p_value_adjusted'] = adjusted_p_values[i]
                stat_results[valid_idx]['significant'] = (adjusted_p_values[i] < 0.05)

            # Print results with interpretation
            print("\n   Effect Size Interpretation Guide:")
            print("     |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, ≥0.8: large")
            print()
            
            for result in stat_results:
                opt_a = result['optimizer_a']
                opt_b = result['optimizer_b']
                p_raw = result['p_value']
                p_adj = result.get('p_value_adjusted', float('nan'))
                cohens_d = result['cohens_d']
                d_ci_low = result.get('cohens_d_ci_lower', float('nan'))
                d_ci_high = result.get('cohens_d_ci_upper', float('nan'))
                interpretation = result.get('effect_interpretation', '')
                sig_mark = "*" if result['significant'] else ""
                
                if not math.isnan(p_adj) and not math.isnan(d_ci_low):
                    print(f"      {opt_a} vs {opt_b}:")
                    print(f"        p={p_raw:.4f}, p_adj={p_adj:.4f}{sig_mark}")
                    print(f"        d={cohens_d:.3f} [95% CI: {d_ci_low:.3f}, {d_ci_high:.3f}] ({interpretation})")
                elif not math.isnan(p_adj):
                    print(f"      {opt_a} vs {opt_b}: p={p_raw:.4f}, p_adj={p_adj:.4f}{sig_mark}, d={cohens_d:.3f}")
                else:
                    print(f"      {opt_a} vs {opt_b}: p={p_raw:.4f} (no adjustment), d={cohens_d:.3f}")
        else:
            # All p-values were NaN
            for result in stat_results:
                result['p_value_adjusted'] = float('nan')
                result['significant'] = False

    # Save statistical results
    if stat_results:
        stat_df = pd.DataFrame(stat_results)
        stat_path = analysis_dir / "cross_experiment_statistics.csv"
        stat_df.to_csv(stat_path, index=False)
        print(f"\n   Cross-experiment statistics saved to {stat_path}")

    return stat_results
