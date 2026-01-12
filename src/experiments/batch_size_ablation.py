#!/usr/bin/env python3
"""
Batch Size Ablation Study

Tests the impact of different batch sizes on optimizer performance across
multiple optimizers to understand scalability and convergence behavior.

Key Questions:
1. How do optimizers perform with different batch sizes?
2. Which optimizers scale best to larger batches?
3. What is the optimal batch size for each optimizer?
4. Is there a batch size-learning rate relationship?
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, cast
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.statistical_analysis import compare_two_optimizers, print_ttest_results


def create_batch_size_configs(
    base_config: Dict,
    batch_sizes: List[int],
    optimizers: List[str],
    apply_lr_scaling: bool = True
) -> List[Dict]:
    """
    Create experiment configurations for batch size ablation.

    SCIENTIFIC VALIDITY WARNING: This is NOT a pure single-variable ablation.
    By default, this function applies Learning Rate Scaling (Linear Scaling Rule)
    which CHANGES BOTH batch size AND learning rate simultaneously.
    This confounds variables and violates ceteris paribus principle.

    Set apply_lr_scaling=False for true single-variable batch size ablation,
    but note that this may lead to unfair comparisons due to scale mismatch.

    Args:
        base_config: Base configuration with model, dataset, etc.
        batch_sizes: List of batch sizes to test
        optimizers: List of optimizer names
        apply_lr_scaling: Whether to scale LR with batch size (default: True)

    Returns:
        List of configuration dictionaries
    """
    configs = []

    # Reference batch size and LR for scaling
    base_batch_size = base_config.get('batch_size', 128)
    base_lr = base_config.get('lr', 0.001)

    for optimizer in optimizers:
        for batch_size in batch_sizes:
            import copy as copy_module
            config = copy_module.deepcopy(base_config)

            # CRITICAL: Apply Linear Scaling Rule for SGD variants
            # For Adam/AdamW, use square root scaling (more conservative)
            if apply_lr_scaling:
                scaling_factor = batch_size / base_batch_size
                if 'Adam' in optimizer or 'RMSprop' in optimizer:
                    # Square root scaling for adaptive optimizers
                    scaled_lr = base_lr * np.sqrt(scaling_factor)
                else:
                    # Linear scaling for SGD variants (Goyal et al., 2017)
                    scaled_lr = base_lr * scaling_factor
            else:
                scaled_lr = base_lr

            config.update({
                'optimizer': optimizer,
                'batch_size': batch_size,
                'lr': scaled_lr,
                'name': f"{optimizer}_batch{batch_size}"
            })
            configs.append(config)

    return configs


def run_batch_size_ablation(
    base_config: Dict,
    batch_sizes: List[int] = [16, 32, 64, 128, 256, 512],
    optimizers: List[str] = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW'],
    seeds: List[int] = [1, 2, 3, 4, 5],
    results_dir: str = 'results/batch_ablation',
    apply_lr_scaling: bool = True  # GAP FIX: Make LR scaling optional
) -> Dict[str, pd.DataFrame]:
    """
    Run batch size ablation study with multiple seeds.

    GAP FIX: Added apply_lr_scaling parameter for scientifically valid single-variable ablation.
    - True (default): Scale LR with batch size (tests practical scalability)
    - False: Keep LR constant (isolates batch size effect for theoretical analysis)

    Args:
        base_config: Base experiment configuration
        batch_sizes: List of batch sizes to test
        optimizers: List of optimizers to compare
        seeds: Random seeds for reproducibility
        results_dir: Output directory
        apply_lr_scaling: Whether to scale learning rate with batch size (default: True)

    Returns:
        Dictionary mapping config names to aggregated results
    """
    os.makedirs(results_dir, exist_ok=True)

    print("="*80)
    print("BATCH SIZE ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {base_config.get('dataset', 'MNIST')}")
    print(f"Model: {base_config.get('model', 'SimpleMLP')}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Optimizers: {optimizers}")
    print(f"Seeds: {seeds}")
    print("="*80)

    # Create configurations with optional LR scaling
    configs = create_batch_size_configs(
        base_config, batch_sizes, optimizers, apply_lr_scaling=apply_lr_scaling
    )

    results = {}

    for config in configs:
        config_name = config['name']
        print(f"\n{'─'*80}")
        print(f"Running: {config_name}")
        print(f"{'─'*80}")

        seed_results = []

        for seed in seeds:
            print(f"  Seed {seed}... ", end='', flush=True)

            # Add seed to config
            import copy as copy_module
            config_with_seed = copy_module.deepcopy(config)
            config_with_seed['seed'] = seed

            # Import and run experiment
            try:
                from src.experiments.run_nn_experiment import train_and_evaluate
                df = train_and_evaluate(config_with_seed)
                # Coerce to DataFrame to satisfy type checker and ensure consistent API
                df = pd.DataFrame(df)

                # Save individual result
                filename = f"{config_name}_seed{seed}.csv"
                filepath = os.path.join(results_dir, filename)
                df.to_csv(filepath, index=False)

                seed_results.append(df)

                # Print final accuracy
                eval_df = df[df['phase'] == 'eval']
                from src.utils.type_guards import ensure_dataframe, ensure_series
                eval_df = ensure_dataframe(eval_df)
                if not eval_df.empty:
                    final_acc = ensure_series(eval_df['test_accuracy']).iloc[-1]
                    print(f"Test Acc: {final_acc:.4f}")
                else:
                    print("Done")

            except Exception as e:
                print(f"Error: {e}")
                continue

        # Aggregate results
        if seed_results:
            results[config_name] = pd.concat(seed_results, ignore_index=True)

    print("\n" + "="*80)
    print("Batch size ablation completed!")
    print("="*80)

    return results


def analyze_batch_size_results(
    results: Dict[str, pd.DataFrame],
    batch_sizes: List[int],
    optimizers: List[str]
) -> pd.DataFrame:
    """
    Analyze batch size ablation results.

    Returns:
        DataFrame with summary statistics
    """
    summary_data = []

    from src.utils.type_guards import ensure_dataframe

    for optimizer in optimizers:
        for batch_size in batch_sizes:
            config_name = f"{optimizer}_batch{batch_size}"

            if config_name not in results:
                continue

            df = ensure_dataframe(results[config_name])

            # Extract final test accuracies from all seeds
            eval_df = ensure_dataframe(df[df['phase'] == 'eval'])

            if eval_df.empty:
                continue

            # Group by seed and get final accuracy
            final_accs = []
            from src.utils.type_guards import ensure_dataframe, ensure_series
            for seed in ensure_series(eval_df['seed']).unique():
                seed_df = ensure_dataframe(eval_df[eval_df['seed'] == seed])
                if not seed_df.empty:
                    # Skip tainted runs
                    if 'tainted' in seed_df.columns and bool(ensure_series(seed_df['tainted']).any()):
                        continue
                    final_accs.append(ensure_series(seed_df['test_accuracy']).iloc[-1])

            if final_accs:
                summary_data.append({
                    'Optimizer': optimizer,
                    'Batch Size': batch_size,
                    'Mean Accuracy': np.mean(final_accs),
                    'Std Accuracy': np.std(final_accs),
                    'Min Accuracy': np.min(final_accs),
                    'Max Accuracy': np.max(final_accs),
                    'N Seeds': len(final_accs)
                })

    summary_df = pd.DataFrame(summary_data)

    return summary_df


def plot_batch_size_trends(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot batch size vs accuracy trends for all optimizers.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    optimizers = summary_df['Optimizer'].unique()
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(optimizers)))

    from src.utils.plot_helpers import arr_to_numpy_float
    for i, optimizer in enumerate(optimizers):
        opt_df = summary_df[summary_df['Optimizer'] == optimizer]
        opt_df = cast(pd.DataFrame, opt_df).sort_values(by=['Batch Size'])

        batch_sizes = arr_to_numpy_float(opt_df['Batch Size'])
        means = arr_to_numpy_float(opt_df['Mean Accuracy'])
        stds = arr_to_numpy_float(opt_df['Std Accuracy'])

        # Plot line with error bars
        ax.errorbar(batch_sizes, means, yerr=stds,
                   marker='o', markersize=8, linewidth=2.5,
                   capsize=5, capthick=2, label=optimizer,
                   color=colors[i], alpha=0.8)

    ax.set_xlabel('Batch Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (Mean ± Std)', fontsize=13, fontweight='bold')
    ax.set_title('Batch Size Ablation: Impact on Optimizer Performance',
                fontsize=15, fontweight='bold', pad=15)

    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"Batch size trend plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_batch_size_heatmap(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot heatmap of accuracy across optimizers and batch sizes.
    """
    # Pivot data for heatmap
    pivot_df = summary_df.pivot(
        index='Optimizer',
        columns='Batch Size',
        values='Mean Accuracy'
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='RdYlGn',
               linewidths=0.5, cbar_kws={'label': 'Mean Test Accuracy'},
               ax=ax, vmin=pivot_df.min().min(), vmax=pivot_df.max().max())

    ax.set_title('Batch Size Ablation Heatmap', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Batch Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Optimizer', fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"Batch size heatmap saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def print_batch_size_summary(summary_df: pd.DataFrame):
    """Print formatted batch size ablation summary."""
    print("\n" + "="*80)
    print("BATCH SIZE ABLATION RESULTS")
    print("="*80)

    optimizers = summary_df['Optimizer'].unique()

    for optimizer in optimizers:
        print(f"\n{optimizer}:")
        print("-" * 80)
        opt_df = summary_df[summary_df['Optimizer'] == optimizer]
        opt_df = cast(pd.DataFrame, opt_df).sort_values(by=['Batch Size'])

        for _, row in opt_df.iterrows():
            print(f"  Batch Size {int(row['Batch Size']):4d}: "
                  f"{row['Mean Accuracy']:.4f} ± {row['Std Accuracy']:.4f} "
                  f"(n={int(row['N Seeds'])})")

        # Find optimal batch size
        best_idx = opt_df['Mean Accuracy'].idxmax()
        best_batch = opt_df.loc[best_idx, 'Batch Size']
        best_acc = opt_df.loc[best_idx, 'Mean Accuracy']
        print(f"  → Optimal: Batch Size {int(best_batch)} ({best_acc:.4f})")

    print("\n" + "="*80)


def perform_batch_size_comparisons(
    results: Dict[str, pd.DataFrame],
    optimizers: List[str],
    batch_sizes: List[int]
):
    """
    Perform statistical comparisons between batch sizes for each optimizer.
    """
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS: Batch Size Effects")
    print("="*80)

    for optimizer in optimizers:
        print(f"\n{optimizer}:")
        print("-" * 80)

        # Compare each batch size against the smallest
        baseline_batch = min(batch_sizes)
        baseline_name = f"{optimizer}_batch{baseline_batch}"

        if baseline_name not in results:
            continue

        baseline_df = results[baseline_name]
        # Ensure DataFrame for safe attribute access
        baseline_df = pd.DataFrame(baseline_df)
        baseline_eval = baseline_df[baseline_df['phase'] == 'eval']

        # Extract final accuracies per seed
        baseline_accs = []
        from src.utils.type_guards import ensure_series, ensure_dataframe
        for seed in ensure_series(baseline_eval['seed']).unique():
            seed_df = ensure_dataframe(baseline_eval[baseline_eval['seed'] == seed])
            if seed_df.shape[0] > 0:
                # Skip tainted seeds
                from src.utils.type_guards import ensure_series
                tainted_any = bool(ensure_series(seed_df['tainted']).any()) if 'tainted' in seed_df.columns else False
                if tainted_any:
                    continue
                baseline_accs.append(ensure_series(seed_df['test_accuracy']).iloc[-1])

        baseline_accs = np.array(baseline_accs)

        for batch_size in batch_sizes:
            if batch_size == baseline_batch:
                continue

            config_name = f"{optimizer}_batch{batch_size}"

            if config_name not in results:
                continue

            df = results[config_name]
            # Ensure DataFrame for safe attribute access
            df = pd.DataFrame(df)
            eval_df = df[df['phase'] == 'eval']

            # Extract final accuracies
            accs = []
            from src.utils.type_guards import ensure_series, ensure_dataframe
            for seed in ensure_series(eval_df['seed']).unique():
                seed_df = ensure_dataframe(eval_df[eval_df['seed'] == seed])
                if seed_df.shape[0] > 0:
                    # Skip tainted seeds
                    from src.utils.type_guards import ensure_series
                    tainted_any = bool(ensure_series(seed_df['tainted']).any()) if 'tainted' in seed_df.columns else False
                    if tainted_any:
                        continue
                    accs.append(ensure_series(seed_df['test_accuracy']).iloc[-1])

            accs = np.array(accs)

            if len(accs) > 0 and len(baseline_accs) > 0:
                result = compare_two_optimizers(
                    accs, baseline_accs,
                    opt1_name=f"Batch {batch_size}",
                    opt2_name=f"Batch {baseline_batch} (baseline)",
                    metric='test_accuracy'
                )

                # Use effect_size field which handles both parametric and non-parametric
                effect_size_val = result.get('effect_size', result.get('cohens_d', 0.0))
                if effect_size_val is None:
                    effect_size_val = 0.0
                effect_size_type = result.get('effect_size_type', 'unknown')

                print(f"\n  Batch {batch_size} vs Batch {baseline_batch}:")
                print(f"    Mean diff: {result['mean_diff']:+.4f}")
                print(f"    p-value: {result['p_value']:.4e}")
                print(f"    Effect size ({effect_size_type}): {effect_size_val:.3f}")
                print(f"    Significant: {'✓' if result['is_significant'] else '✗'}")


def main():
    """Run full batch size ablation study."""

    # Base configuration
    base_config = {
        'dataset': 'MNIST',
        'model': 'SimpleMLP',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 10
    }

    # Test configurations
    batch_sizes = [16, 32, 64, 128, 256, 512]
    optimizers = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW']
    seeds = [1, 2, 3, 4, 5]

    # Run ablation study
    results = run_batch_size_ablation(
        base_config,
        batch_sizes=batch_sizes,
        optimizers=optimizers,
        seeds=seeds
    )

    # Analyze results
    summary_df = analyze_batch_size_results(results, batch_sizes, optimizers)

    # Print summary
    print_batch_size_summary(summary_df)

    # Save summary
    os.makedirs('results/batch_ablation', exist_ok=True)
    summary_df.to_csv('results/batch_ablation/batch_size_summary.csv', index=False)
    print("\nSummary saved to: results/batch_ablation/batch_size_summary.csv")

    # Create visualizations
    os.makedirs('plots', exist_ok=True)
    plot_batch_size_trends(summary_df, save_path='plots/batch_size_trends.png')
    plot_batch_size_heatmap(summary_df, save_path='plots/batch_size_heatmap.png')

    # Statistical comparisons
    perform_batch_size_comparisons(results, optimizers, batch_sizes)

    print("\n" + "="*80)
    print("BATCH SIZE ABLATION STUDY COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
