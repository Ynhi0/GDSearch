#!/usr/bin/env python3
"""
Weight Decay Ablation Study

Systematically tests different weight decay values across optimizers to:
1. Understand regularization impact on generalization
2. Find optimal weight decay for each optimizer
3. Compare optimizer sensitivity to weight decay
4. Analyze train/test gap (generalization)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from src.utils.plot_helpers import arr_to_numpy_float
from src.utils.type_guards import ensure_dataframe

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.num_utils import safe_to_float
from src.utils.file_safety import safe_to_csv


def create_weight_decay_configs(
    base_config: Dict,
    weight_decays: List[float],
    optimizers: List[str],
    exclude_bn_from_wd: bool = True
) -> List[Dict]:
    """
    Create experiment configurations for weight decay ablation.

    GAP #14 FIX: Added exclude_bn_from_wd parameter (default=True)
    Scientific Rationale: Batch Normalization is scale-invariant, so weight decay on BN
    parameters does NOT provide regularization - it only increases the effective learning rate.
    For fair comparison, WD should only apply to convolutional/linear layer weights.

    Args:
        base_config: Base configuration with model, dataset, etc.
        weight_decays: List of weight decay values to test
        optimizers: List of optimizer names
        exclude_bn_from_wd: If True, exclude BatchNorm parameters from weight decay (RECOMMENDED)

    Returns:
        List of configuration dictionaries
    """
    configs = []

    for optimizer in optimizers:
        for wd in weight_decays:
            import copy as copy_module
            config = copy_module.deepcopy(base_config)
            config.update({
                'optimizer': optimizer,
                'weight_decay': wd,
                'exclude_bn_from_wd': exclude_bn_from_wd,  # GAP #14 FIX
                'name': f"{optimizer}_wd{wd:.1e}"
            })
            configs.append(config)

    return configs


def run_weight_decay_ablation(
    base_config: Dict,
    weight_decays: Optional[List[float]] = None,
    optimizers: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    results_dir: str = 'results/wd_ablation'
) -> Dict[str, pd.DataFrame]:
    """
    Run weight decay ablation study with multiple seeds.

    Args:
        base_config: Base experiment configuration
        weight_decays: List of weight decay values to test (default: [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        optimizers: List of optimizer names (default: ['SGD', 'SGD_Momentum', 'Adam', 'AdamW'])
        seeds: List of random seeds (default: [1, 2, 3, 4, 5])
        results_dir: Directory to save results

    Returns:
        Dictionary of results DataFrames
    """
    # Set defaults
    if weight_decays is None:
        weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    if optimizers is None:
        optimizers = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW']
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    os.makedirs(results_dir, exist_ok=True)

    print("="*80)
    print("WEIGHT DECAY ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {base_config.get('dataset', 'MNIST')}")
    print(f"Model: {base_config.get('model', 'SimpleMLP')}")
    print(f"Weight decays: {weight_decays}")
    print(f"Optimizers: {optimizers}")
    print(f"Seeds: {seeds}")
    print("="*80)

    # Create configurations
    configs = create_weight_decay_configs(base_config, weight_decays, optimizers)

    results = {}

    for config in configs:
        config_name = config['name']
        print(f"\n{'─'*80}")
        print(f"Running: {config_name}")
        print(f"{'─'*80}")

        seed_results = []

        for seed in seeds:
            print(f"  Seed {seed}... ", end='', flush=True)

            # Add seed to config (use deepcopy to prevent mutation leakage)
            import copy as copy_module
            config_with_seed = copy_module.deepcopy(config)
            config_with_seed['seed'] = seed

            # Import and run experiment
            try:
                from src.experiments.run_nn_experiment import train_and_evaluate
                df = train_and_evaluate(config_with_seed)
                df = pd.DataFrame(df)
                df = ensure_dataframe(df)
                assert isinstance(df, pd.DataFrame)

                # Save individual result
                filename = f"{config_name}_seed{seed}.csv"
                filepath = os.path.join(results_dir, filename)
                safe_to_csv(df, filepath, index=False)  # AUDIT FIX: Safe I/O

                seed_results.append(df)

                # Print final metrics (for debugging/monitoring)
                eval_df = df[df['phase'] == 'eval']
                eval_df = ensure_dataframe(eval_df)
                if not eval_df.empty:
                    from src.utils.type_guards import ensure_series
                    final_acc = safe_to_float(ensure_series(eval_df['test_accuracy']).iloc[-1])
                    final_loss = safe_to_float(ensure_series(eval_df['test_loss']).iloc[-1])
                    print(f"Acc: {final_acc:.4f}, Loss: {final_loss:.4f}")
                else:
                    print("Done")

            except (RuntimeError, ValueError, OSError, KeyError) as e:
                print(f"Error: {e}")
                continue

        if seed_results:
            results[config_name] = pd.concat(seed_results, ignore_index=True)

    print("\n" + "="*80)
    print("Weight decay ablation completed!")
    print("="*80)

    return results


def analyze_weight_decay_results(
    results: Dict[str, pd.DataFrame],
    weight_decays: List[float],
    optimizers: List[str]
) -> pd.DataFrame:
    """
    Analyze weight decay ablation results.

    Returns:
        DataFrame with summary statistics including generalization gap
    """
    summary_data = []

    for optimizer in optimizers:
        for wd in weight_decays:
            config_name = f"{optimizer}_wd{wd:.1e}"

            if config_name not in results:
                continue

            from src.utils.type_guards import ensure_dataframe
            df = ensure_dataframe(results[config_name])

            # Extract final test accuracies from all seeds
            eval_df = ensure_dataframe(df[df['phase'] == 'eval'])

            if eval_df.empty:
                continue

            # Group by seed and get final metrics
            final_test_accs = []
            final_test_losses = []
            gen_gaps = []

            from src.utils.type_guards import ensure_series, ensure_dataframe
            for seed in ensure_series(eval_df['seed']).unique():
                seed_eval = ensure_dataframe(eval_df[eval_df['seed'] == seed])
                if seed_eval.empty:
                    continue
                # Skip tainted seeds (explicit bool conversion for Series.any())
                tainted_any = False
                if 'tainted' in seed_eval.columns:
                    tainted_any = bool(ensure_series(seed_eval['tainted']).any())
                if tainted_any:
                    continue

                final_test_accs.append(safe_to_float(ensure_series(seed_eval['test_accuracy']).iloc[-1]))
                final_test_losses.append(safe_to_float(ensure_series(seed_eval['test_loss']).iloc[-1]))

                # Calculate generalization gap (test_loss - train_loss)
                train_df = ensure_dataframe(df[(df['phase'] == 'train') & (df['seed'] == seed)])
                if not train_df.empty:
                    train_loss = safe_to_float(ensure_series(train_df['train_loss']).iloc[-1])
                    test_loss = final_test_losses[-1]
                    gen_gaps.append(float(test_loss - train_loss))

            if final_test_accs:
                summary_data.append({
                    'Optimizer': optimizer,
                    'Weight Decay': wd,
                    'Mean Accuracy': np.mean(final_test_accs),
                    'Std Accuracy': np.std(final_test_accs),
                    'Mean Test Loss': np.mean(final_test_losses),
                    'Mean Gen Gap': np.mean(gen_gaps) if gen_gaps else np.nan,
                    'N Seeds': len(final_test_accs)
                })

    summary_df = pd.DataFrame(summary_data)

    return summary_df


def plot_weight_decay_trends(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot weight decay vs accuracy and generalization gap.
    """
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    from matplotlib import cm
    optimizers = summary_df['Optimizer'].unique()
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(optimizers)))

    # Plot 1: Accuracy
    for i, optimizer in enumerate(optimizers):
        opt_df = summary_df[summary_df['Optimizer'] == optimizer]
        opt_df = ensure_dataframe(opt_df).sort_values(by=['Weight Decay'])

        wds = arr_to_numpy_float(opt_df['Weight Decay'])
        means = arr_to_numpy_float(opt_df['Mean Accuracy'])
        stds = arr_to_numpy_float(opt_df['Std Accuracy'])

        ax1.errorbar(wds, means, yerr=stds,
                    marker='o', markersize=8, linewidth=2.5,
                    capsize=5, capthick=2, label=optimizer,
                    color=colors[i], alpha=0.8)

        # Mark optimal weight decay
        best_idx = opt_df['Mean Accuracy'].idxmax()
        best_wd = opt_df.loc[best_idx, 'Weight Decay']
        best_acc = opt_df.loc[best_idx, 'Mean Accuracy']
        # Convert to float for comparison using safe_to_float
        best_wd_float = safe_to_float(best_wd)
        ax1.plot(best_wd_float if best_wd_float > 0 else 1e-7, best_acc, 'r*', markersize=15,
                markeredgecolor='black', markeredgewidth=1.5)

    ax1.set_xlabel('Weight Decay', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (Mean ± Std)', fontsize=13, fontweight='bold')
    ax1.set_title('Weight Decay Impact on Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xscale('symlog', linthresh=1e-6)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)

    # Plot 2: Generalization Gap
    for i, optimizer in enumerate(optimizers):
        opt_df = summary_df[summary_df['Optimizer'] == optimizer]
        opt_df = ensure_dataframe(opt_df).sort_values(by=['Weight Decay'])

        wds = arr_to_numpy_float(opt_df['Weight Decay'])
        gaps = arr_to_numpy_float(opt_df['Mean Gen Gap'])

        ax2.plot(wds, gaps, marker='s', markersize=8, linewidth=2.5,
                label=optimizer, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Weight Decay', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Generalization Gap (Test - Train Loss)', fontsize=13, fontweight='bold')
    ax2.set_title('Weight Decay Impact on Generalization', fontsize=14, fontweight='bold')
    ax2.set_xscale('symlog', linthresh=1e-6)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect Generalization')

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"Weight decay trends plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def print_weight_decay_summary(summary_df: pd.DataFrame):
    """Print formatted weight decay ablation summary."""
    print("\n" + "="*80)
    print("WEIGHT DECAY ABLATION RESULTS")
    print("="*80)

    optimizers = summary_df['Optimizer'].unique()

    for optimizer in optimizers:
        print(f"\n{optimizer}:")
        print("-" * 80)
        opt_df = summary_df[summary_df['Optimizer'] == optimizer]
        opt_df = ensure_dataframe(opt_df).sort_values(by=['Weight Decay'])

        for _, row in opt_df.iterrows():
            wd_val = safe_to_float(row['Weight Decay'])
            wd_str = f"{wd_val:.1e}" if wd_val > 0 else "0.0    "
            print(f"  WD {wd_str}: "
                  f"Acc={row['Mean Accuracy']:.4f}±{row['Std Accuracy']:.4f}, "
                  f"GenGap={row['Mean Gen Gap']:.4f}")

        # Find optimal weight decay (best accuracy with lowest gen gap)
        best_idx = opt_df['Mean Accuracy'].idxmax()
        best_wd = safe_to_float(opt_df.loc[best_idx, 'Weight Decay'])
        best_acc = opt_df.loc[best_idx, 'Mean Accuracy']
        best_gap = opt_df.loc[best_idx, 'Mean Gen Gap']
        print(f"  → Optimal: WD {best_wd:.1e} (Acc={best_acc:.4f}, GenGap={best_gap:.4f})")

    print("\n" + "="*80)


def main():
    """Run full weight decay ablation study."""

    # Base configuration
    base_config = {
        'dataset': 'MNIST',
        'model': 'SimpleMLP',
        'lr': 1e-3,
        'epochs': 10,
        'batch_size': 128
    }

    # Test configurations
    weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    optimizers = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW']
    seeds = [1, 2, 3, 4, 5]

    # Run ablation study
    results = run_weight_decay_ablation(
        base_config,
        weight_decays=weight_decays,
        optimizers=optimizers,
        seeds=seeds
    )

    # Analyze results
    summary_df = analyze_weight_decay_results(results, weight_decays, optimizers)

    # Print summary
    print_weight_decay_summary(summary_df)

    # Save summary
    os.makedirs('results/wd_ablation', exist_ok=True)
    safe_to_csv(summary_df, 'results/wd_ablation/weight_decay_summary.csv', index=False)
    print("\nSummary saved to: results/wd_ablation/weight_decay_summary.csv")

    # Create visualizations
    os.makedirs('plots', exist_ok=True)
    plot_weight_decay_trends(summary_df, save_path='plots/weight_decay_trends.png')

    print("\n" + "="*80)
    print("WEIGHT DECAY ABLATION STUDY COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
