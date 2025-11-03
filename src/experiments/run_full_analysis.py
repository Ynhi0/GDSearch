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
from src.analysis.statistical_analysis import compare_optimizers_ttest, print_ttest_results, plot_comparison_with_errorbars
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
            
            finals_A = []
            finals_B = []
            
            for df in dfs_A:
                if task_type == 'neural_network':
                    eval_df = df[df['phase'] == 'eval']
                    if not eval_df.empty:
                        finals_A.append(eval_df[metric].iloc[-1])
                else:
                    finals_A.append(df[metric].iloc[-1])
            
            for df in dfs_B:
                if task_type == 'neural_network':
                    eval_df = df[df['phase'] == 'eval']
                    if not eval_df.empty:
                        finals_B.append(eval_df[metric].iloc[-1])
                else:
                    finals_B.append(df[metric].iloc[-1])
            
            results_A = np.array(finals_A)
            results_B = np.array(finals_B)
            
            # Perform t-test
            result = compare_optimizers_ttest(
                results_A, results_B,
                name_A=opt_A,
                name_B=opt_B,
                metric=metric
            )
            
            print_ttest_results(result)
            
            # Plot comparison
            plot_path = os.path.join(plots_dir, f'statistical_{opt_A}_vs_{opt_B}.png')
            plot_comparison_with_errorbars(
                results_A, results_B,
                name_A=opt_A,
                name_B=opt_B,
                metric=metric.replace('_', ' ').title(),
                save_path=plot_path
            )
    
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
        # Assume format: OptimizerName_..._seedX.csv
        opt_name = filename.split('_')[0]
        
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
    
    parser.add_argument('--config', type=str, default='configs/mnist_tuning.json',
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
