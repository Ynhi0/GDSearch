"""
Full multi-seed experiment and statistical analysis pipeline.

Usage:
    python run_full_analysis.py --dataset mnist --seeds 1,2,3,4,5
"""

import os
import argparse
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict

from src.experiments.run_multi_seed import run_multi_seed_experiment, aggregate_results, save_aggregated_results
from src.analysis.statistical_analysis import (
    compare_optimizers_ttest,
    print_ttest_results,
    plot_comparison_with_errorbars,
    auto_select_test,
    holm_bonferroni_correction,
    power_analysis_report,
)
from src.visualization.plot_results import plot_multiseed_comparison, plot_final_metric_comparison


def load_multiseed_results(pattern: str, results_dir: str = 'results') -> List[pd.DataFrame]:
    """Load all CSVs matching a pattern."""
    files = glob.glob(os.path.join(results_dir, pattern))
    return [pd.read_csv(f) for f in sorted(files)]


def run_full_pipeline(
    config_path: str,
    seeds: List[int],
    results_dir: str = 'results',
    plots_dir: str = 'plots',
    comparison_pairs: List[tuple] = None
):
    """
    Run full multi-seed experiment pipeline:
    1. Run experiments with multiple seeds
    2. Aggregate results
    3. Statistical comparison
    4. Generate plots with error bars
    
    Args:
        config_path: Path to base config file
        seeds: List of random seeds
        results_dir: Directory for results
        plots_dir: Directory for plots
        comparison_pairs: List of (optimizer_A, optimizer_B) tuples to compare
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    print("="*70)
    print(f"FULL MULTI-SEED ANALYSIS PIPELINE")
    print("="*70)
    print(f"Config: {config_path}")
    print(f"Seeds: {seeds}")
    print(f"Number of seeds: {len(seeds)}")
    print(f"Results dir: {results_dir}")
    print(f"Plots dir: {plots_dir}")
    print("="*70)
    
    # ====================================================================
    # PHASE 1: RUN MULTI-SEED EXPERIMENTS
    # ====================================================================
    print("\n[PHASE 1] Running multi-seed experiments...")
    print("-"*70)
    
    result_files = run_multi_seed_experiment(
        base_config=base_config,
        seeds=seeds,
        results_dir=results_dir
    )
    
    print(f"✅ Completed {len(result_files)} experiment runs")
    
    # ====================================================================
    # PHASE 2: AGGREGATE RESULTS
    # ====================================================================
    print("\n[PHASE 2] Aggregating results...")
    print("-"*70)
    
    # Determine metric based on task type
    task_type = base_config.get('task', 'neural_network')
    if task_type == 'neural_network':
        metrics = ['test_accuracy', 'test_loss', 'train_loss']
    else:
        metrics = ['loss', 'grad_norm', 'update_norm']
    
    aggregated = {}
    for metric in metrics:
        agg = aggregate_results(result_files, metric=metric)
        aggregated[metric] = agg
        
        # Print summary
        print(f"\n{metric}:")
        print(f"  Mean: {agg['mean']:.4f}")
        print(f"  Std:  {agg['std']:.4f}")
        print(f"  Min:  {agg['min']:.4f}")
        print(f"  Max:  {agg['max']:.4f}")
    
    # Save aggregated results
    agg_path = os.path.join(results_dir, 'aggregated_results.json')
    save_aggregated_results(aggregated, agg_path)
    print(f"\n✅ Aggregated results saved to: {agg_path}")
    
    # ====================================================================
    # PHASE 3: STATISTICAL COMPARISONS
    # ====================================================================
    if comparison_pairs:
        print("\n[PHASE 3] Statistical comparisons...")
        print("-"*70)
        summary_rows = []
        p_values = []
        
        for opt_A, opt_B in comparison_pairs:
            print(f"\n{'='*70}")
            print(f"Comparing: {opt_A} vs {opt_B}")
            print(f"{'='*70}")
            
            # Load results for both optimizers
            pattern_A = f"*{opt_A}*seed*.csv"
            pattern_B = f"*{opt_B}*seed*.csv"
            
            dfs_A = load_multiseed_results(pattern_A, results_dir)
            dfs_B = load_multiseed_results(pattern_B, results_dir)
            
            if not dfs_A or not dfs_B:
                print(f"⚠️ Missing results for {opt_A} or {opt_B}, skipping...")
                continue
            
            # Extract final metric
            metric = metrics[0]  # Primary metric (test_accuracy or loss)
            
            # Build final metrics and attempt to enable paired comparison by matching seeds
            def _extract_seed(fname: str):
                import re, os
                m = re.search(r"seed(\d+)", os.path.basename(fname))
                return int(m.group(1)) if m else None

            paired = False
            if task_type == 'neural_network':
                files_A = glob.glob(os.path.join(results_dir, pattern_A))
                files_B = glob.glob(os.path.join(results_dir, pattern_B))
                map_A, map_B = {}, {}
                for f in files_A:
                    s = _extract_seed(f)
                    if s is None:
                        continue
                    try:
                        df = pd.read_csv(f)
                        ev = df[df['phase'] == 'eval']
                        if not ev.empty:
                            map_A[s] = float(ev[metric].iloc[-1])
                    except Exception:
                        pass
                for f in files_B:
                    s = _extract_seed(f)
                    if s is None:
                        continue
                    try:
                        df = pd.read_csv(f)
                        ev = df[df['phase'] == 'eval']
                        if not ev.empty:
                            map_B[s] = float(ev[metric].iloc[-1])
                    except Exception:
                        pass
                common = sorted(set(map_A) & set(map_B))
                seeds_A_only = set(map_A) - set(map_B)
                seeds_B_only = set(map_B) - set(map_A)
                n_common_seeds = len(common)
                
                # Warn if seeds don't fully match
                if seeds_A_only or seeds_B_only:
                    print(f"⚠️  Seed mismatch detected:")
                    print(f"   {opt_A} has {len(map_A)} seeds, {opt_B} has {len(map_B)} seeds, {n_common_seeds} in common")
                    if seeds_A_only:
                        print(f"   {opt_A}-only seeds: {sorted(seeds_A_only)}")
                    if seeds_B_only:
                        print(f"   {opt_B}-only seeds: {sorted(seeds_B_only)}")
                    print(f"   Recommendation: Use identical seed lists for all optimizers to enable pairing.\n")
                
                if n_common_seeds >= 3:
                    results_A = np.array([map_A[s] for s in common], dtype=float)
                    results_B = np.array([map_B[s] for s in common], dtype=float)
                    paired = True
                    print(f"✅ Using paired test with {n_common_seeds} common seeds: {common}\n")
                else:
                    # Fallback to unpaired if insufficient common seeds
                    if n_common_seeds > 0:
                        print(f"⚠️  Only {n_common_seeds} common seed(s), need ≥3 for pairing. Falling back to unpaired.\n")
                    finals_A = []
                    finals_B = []
                    for df in dfs_A:
                        ev = df[df['phase'] == 'eval']
                        if not ev.empty:
                            finals_A.append(ev[metric].iloc[-1])
                    for df in dfs_B:
                        ev = df[df['phase'] == 'eval']
                        if not ev.empty:
                            finals_B.append(ev[metric].iloc[-1])
                    results_A = np.array(finals_A, dtype=float)
                    results_B = np.array(finals_B, dtype=float)
                    n_common_seeds = 0  # reset for unpaired fallback
            else:
                # 2D tasks: no per-seed files; treat as unpaired
                finals_A = [df[metric].iloc[-1] for df in dfs_A]
                finals_B = [df[metric].iloc[-1] for df in dfs_B]
                results_A = np.array(finals_A, dtype=float)
                results_B = np.array(finals_B, dtype=float)
                n_common_seeds = 0  # 2D experiments typically not seeded in filename

            # Auto-select appropriate test based on normality and pairing
            auto = auto_select_test(results_A, results_B, paired=paired, name_A=opt_A, name_B=opt_B)
            test_result = auto['test_result']
            test_type = auto.get('test_type', 'unknown')
            
            # Unified access to p-value and effect size
            if 'p_value' in test_result:
                p_value = float(test_result['p_value'])
            else:
                # Fallback to parametric
                tt = compare_optimizers_ttest(results_A, results_B, name_A=opt_A, name_B=opt_B, metric=metric)
                p_value = float(tt['p_value'])
                test_result = tt
                test_type = 'parametric (t-test)'
            p_values.append(p_value)
            
            # Print a concise report
            print(f"Test selected: {test_type}")
            if 't_statistic' in test_result:
                print_ttest_results(test_result)
            else:
                # Minimal print for non-param; defer full formatting to CSV
                print(f"p={p_value:.4f}, significant={test_result.get('significant')}")
            
            # Collect summary row
            row = {
                'Optimizer A': opt_A,
                'Optimizer B': opt_B,
                'Metric': metric,
                'Test': test_type,
                'Paired': paired,
                'n_common_seeds': n_common_seeds if paired else 0,
                'p_value': p_value,
                'Significant (raw)': bool(test_result.get('significant', p_value < 0.05)),
            }
            # Effect sizes (standardized columns)
            if 'cohens_d' in test_result:
                row['Effect size'] = float(test_result['cohens_d'])
                row['Effect size type'] = 'Cohen_d'
                row['Effect size (Cohen_d)'] = float(test_result['cohens_d'])  # backward-compatible
            if 'effect_size_r' in test_result:
                row['Effect size'] = float(test_result['effect_size_r'])
                row['Effect size type'] = 'rank_biserial_r'
                row['Effect size (r)'] = float(test_result['effect_size_r'])  # backward-compatible
            # Means (if available)
            for k in ('mean_A','std_A','n_A','mean_B','std_B','n_B'):
                if k in test_result:
                    row[k] = test_result[k]

            # Normality diagnostics from auto-selection
            normA = auto.get('normality_A', {})
            normB = auto.get('normality_B', {})
            row['Normality_A_method'] = normA.get('method')
            row['Normality_A_p'] = normA.get('p_value')
            row['Normality_A_normal'] = normA.get('normal')
            row['Normality_B_method'] = normB.get('method')
            row['Normality_B_p'] = normB.get('p_value')
            row['Normality_B_normal'] = normB.get('normal')

            # Power analysis (based on raw arrays)
            try:
                pow_rep = power_analysis_report(results_A, results_B, name_A=opt_A, name_B=opt_B)
                row['Observed power'] = float(pow_rep.get('achieved_power', np.nan))
                rn = pow_rep.get('required_n', np.nan)
                row['Required n (80% power)'] = int(rn) if isinstance(rn, (int, np.integer)) else (int(rn) if np.isfinite(rn) else np.nan)
            except Exception:
                row['Observed power'] = np.nan
                row['Required n (80% power)'] = np.nan

            summary_rows.append(row)
            
            # Plot comparison
            plot_path = os.path.join(plots_dir, f'statistical_{opt_A}_vs_{opt_B}.png')
            plot_comparison_with_errorbars(
                results_A, results_B,
                name_A=opt_A,
                name_B=opt_B,
                metric=metric.replace('_', ' ').title(),
                save_path=plot_path
            )
        
        # Save CSV summary with Holm-Bonferroni correction across all pairs
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            try:
                signif_corrected = holm_bonferroni_correction(df_summary['p_value'].tolist(), alpha=0.05)
                df_summary['Significant (Holm-Bonferroni)'] = signif_corrected
                correction = 'Holm-Bonferroni'
            except Exception:
                df_summary['Significant (Holm-Bonferroni)'] = np.nan
                correction = 'Unavailable'
            out_csv = os.path.join(results_dir, 'statistical_comparisons.csv')
            df_summary.to_csv(out_csv, index=False)
            print(f"\nSaved statistical comparison summary ({correction}) to: {out_csv}")
            
            # Summary warnings for seed consistency across all comparisons
            paired_count = df_summary['Paired'].sum()
            unpaired_count = len(df_summary) - paired_count
            if unpaired_count > 0 and task_type == 'neural_network':
                print(f"\n⚠️  Seed Consistency Summary:")
                print(f"   {paired_count}/{len(df_summary)} comparisons used paired tests.")
                print(f"   {unpaired_count}/{len(df_summary)} fell back to unpaired (insufficient common seeds).")
                if df_summary['n_common_seeds'].notna().any():
                    min_common = df_summary.loc[df_summary['Paired'], 'n_common_seeds'].min() if paired_count > 0 else 0
                    max_common = df_summary.loc[df_summary['Paired'], 'n_common_seeds'].max() if paired_count > 0 else 0
                    print(f"   Common seeds ranged from {int(min_common)} to {int(max_common)} for paired comparisons.")
                print(f"   Recommendation: Run all optimizers with identical seed lists (≥5 seeds) for full pairing benefit.\n")
    
    # ====================================================================
    # PHASE 4: GENERATE PLOTS WITH ERROR BARS
    # ====================================================================
    print("\n[PHASE 4] Generating plots with error bars...")
    print("-"*70)
    
    # Group results by optimizer
    optimizer_results = {}
    
    for result_file in result_files:
        # Extract optimizer name from filename
        filename = os.path.basename(result_file)
        parts = filename.split('_')
        # NN results: NN_<model>_<dataset>_<optimizer>_...
        if filename.startswith('NN_') and len(parts) >= 4:
            opt_name = parts[3]
        else:
            # 2D results often start with OptimizerName_...
            opt_name = parts[0]

        if opt_name not in optimizer_results:
            optimizer_results[opt_name] = []

        optimizer_results[opt_name].append(pd.read_csv(result_file))
    
    print(f"Found {len(optimizer_results)} optimizers: {list(optimizer_results.keys())}")
    
    # Plot comparison curves with error bars for each metric
    for metric in metrics:
        plot_path = os.path.join(plots_dir, f'multiseed_comparison_{metric}.png')
        
        plot_multiseed_comparison(
            optimizer_results,
            metric=metric,
            title=f'{metric.replace("_", " ").title()} - Multi-Seed Comparison',
            save_path=plot_path
        )
    
    # Plot final metric comparison (bar plot with error bars)
    if task_type == 'neural_network':
        primary_metric = 'test_accuracy'
    else:
        primary_metric = 'loss'
    
    plot_path = os.path.join(plots_dir, f'final_{primary_metric}_comparison.png')
    
    plot_final_metric_comparison(
        optimizer_results,
        metric=primary_metric,
        title=f'Final {primary_metric.replace("_", " ").title()} Comparison',
        save_path=plot_path
    )
    
    print("\n" + "="*70)
    print("✅ FULL PIPELINE COMPLETED!")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"Aggregated results: {agg_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Run full multi-seed analysis pipeline')
    
    parser.add_argument('--config', type=str, default='configs/nn_tuning.json',
                        help='Path to configuration file')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5',
                        help='Comma-separated list of random seeds')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--compare', type=str, default=None,
                        help='Comma-separated pairs to compare, e.g., "AdamW-SGDMomentum,Adam-RMSProp"')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Parse comparison pairs
    comparison_pairs = None
    if args.compare:
        pairs = args.compare.split(',')
        comparison_pairs = [tuple(p.split('-')) for p in pairs]
    
    # Before running, perform a quick schema check for config expectations
    try:
        with open(args.config, 'r') as f:
            base_cfg_preview = json.load(f)
        # If the config looks like a tuning config (has 'sweeps'/'final') but no top-level 'optimizer',
        # provide a helpful message instead of failing deep inside.
        if 'optimizer' not in base_cfg_preview and ('sweeps' in base_cfg_preview or 'final' in base_cfg_preview):
            print("\n⚠️ The provided config appears to be a tuning spec (contains 'sweeps'/'final') without a top-level 'optimizer'.")
            print("   This runner expects a single-optimizer config, e.g.: {\n     'model': 'SimpleMLP', 'dataset': 'MNIST', 'optimizer': 'AdamW', 'lr': 1e-3, 'epochs': 5, 'batch_size': 128\n   }")
            print("   To generate tuned configs and results, run: python scripts/tune_nn.py (which reads configs/nn_tuning.json).\n")
    except Exception:
        # Non-fatal; proceed and let downstream raise if truly invalid
        pass

    # Run pipeline
    run_full_pipeline(
        config_path=args.config,
        seeds=seeds,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        comparison_pairs=comparison_pairs
    )


if __name__ == '__main__':
    main()
