"""
Ablation Study: Component-wise isolation of optimizer features.

This script tests each optimizer component in isolation to quantify
their individual contributions to performance.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from src.experiments.run_nn_experiment import train_and_evaluate
from src.analysis.statistical_analysis import compare_optimizers_ttest, print_ttest_results


def create_ablation_configs(base_config: Dict) -> Dict[str, Dict]:
    """
    Create ablation configurations for Adam/AdamW components.
    
    Components to test:
    1. SGD (baseline - no momentum, no adaptive LR)
    2. SGD + Momentum only
    3. SGD + Adaptive LR only (RMSProp-style)
    4. SGD + Momentum + Adaptive LR (Adam without bias correction)
    5. Adam (full with bias correction)
    6. AdamW (full with decoupled weight decay)
    """
    
    lr = base_config.get('lr', 1e-3)
    weight_decay = base_config.get('weight_decay', 1e-4)
    epochs = base_config.get('epochs', 10)
    batch_size = base_config.get('batch_size', 128)
    dataset = base_config.get('dataset', 'MNIST')
    model = base_config.get('model', 'SimpleMLP')
    
    configs = {
        '1_SGD_baseline': {
            'model': model,
            'dataset': dataset,
            'optimizer': 'SGD',
            'lr': lr,
            'weight_decay': 0.0,
            'momentum': 0.0,
            'epochs': epochs,
            'batch_size': batch_size
        },
        
        '2_SGD_Momentum': {
            'model': model,
            'dataset': dataset,
            'optimizer': 'SGD_Momentum',
            'lr': lr,
            'weight_decay': 0.0,
            'momentum': 0.9,
            'epochs': epochs,
            'batch_size': batch_size
        },
        
        '3_RMSProp_AdaptiveLR': {
            'model': model,
            'dataset': dataset,
            'optimizer': 'RMSProp',
            'lr': lr,
            'weight_decay': 0.0,
            'alpha': 0.99,
            'epochs': epochs,
            'batch_size': batch_size
        },
        
        '4_Adam_no_weight_decay': {
            'model': model,
            'dataset': dataset,
            'optimizer': 'Adam',
            'lr': lr,
            'weight_decay': 0.0,
            'beta1': 0.9,
            'beta2': 0.999,
            'epochs': epochs,
            'batch_size': batch_size
        },
        
        '5_Adam_with_L2': {
            'model': model,
            'dataset': dataset,
            'optimizer': 'Adam',
            'lr': lr,
            'weight_decay': weight_decay,
            'beta1': 0.9,
            'beta2': 0.999,
            'epochs': epochs,
            'batch_size': batch_size
        },
        
        '6_AdamW_decoupled': {
            'model': model,
            'dataset': dataset,
            'optimizer': 'AdamW',
            'lr': lr,
            'weight_decay': weight_decay,
            'beta1': 0.9,
            'beta2': 0.999,
            'epochs': epochs,
            'batch_size': batch_size
        }
    }
    
    return configs


def run_ablation_study(
    base_config: Dict,
    seeds: List[int],
    results_dir: str = 'results/ablation'
) -> Dict[str, List[pd.DataFrame]]:
    """
    Run ablation study with multiple seeds.
    
    Returns:
        Dictionary mapping config name to list of result DataFrames
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create ablation configs
    ablation_configs = create_ablation_configs(base_config)
    
    print("="*70)
    logging.info("ABLATION STUDY: Component-wise Analysis")
    print("="*70)
    logging.info(f"Base config: {base_config.get('dataset')} - {base_config.get('model')}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Configurations: {len(ablation_configs)}")
    print("="*70)
    
    results = {}
    
    for config_name, config in ablation_configs.items():
        logging.info(f"\n{'─'*70}")
        logging.info(f"Running: {config_name}")
        logging.info(f"{'─'*70}")
        
        results[config_name] = []
        
        for seed in seeds:
            print(f"  Seed {seed}... ", end='', flush=True)
            
            config_with_seed = config.copy()
            config_with_seed['seed'] = seed
            
            # Run experiment
            df = train_and_evaluate(config_with_seed)
            
            # Save result
            filename = f"{config_name}_seed{seed}.csv"
            filepath = os.path.join(results_dir, filename)
            df.to_csv(filepath, index=False)
            
            results[config_name].append(df)
            
            # Get final test accuracy
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                from src.utils.type_guards import ensure_series
                final_acc = ensure_series(eval_df['test_accuracy']).iloc[-1]
                logging.info(f"Test Acc: {final_acc:.4f}")
            else:
                logging.info("Done")
    
    print("\n" + "="*70)
    logging.info("Ablation study completed!")
    print("="*70)
    
    return results


def analyze_ablation_results(results: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    """
    Analyze ablation results and compute statistics.
    
    Returns:
        DataFrame with summary statistics for each configuration
    """
    summary_data = []
    
    for config_name, dfs in results.items():
        # Extract final test accuracies
        final_accs = []
        from src.utils.type_guards import ensure_series
        for df in dfs:
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                final_accs.append(ensure_series(eval_df['test_accuracy']).iloc[-1])
        
        if final_accs:
            summary_data.append({
                'Configuration': config_name.replace('_', ' '),
                'Mean Accuracy': np.mean(final_accs),
                'Std Accuracy': np.std(final_accs),
                'Min Accuracy': np.min(final_accs),
                'Max Accuracy': np.max(final_accs),
                'N Seeds': len(final_accs)
            })
    
    summary_df = pd.DataFrame(summary_data)
    from typing import cast
    summary_df = cast(pd.DataFrame, summary_df).sort_values(by=['Mean Accuracy'], ascending=False)
    
    return summary_df


def print_ablation_summary(summary_df: pd.DataFrame):
    """Print formatted ablation study summary."""
    print("\n" + "="*70)
    logging.info("ABLATION STUDY RESULTS")
    print("="*70)
    
    
    for idx, row in summary_df.iterrows():
        logging.info(f"{row['Configuration']}")
        logging.info(f"  Mean: {row['Mean Accuracy']:.4f} ± {row['Std Accuracy']:.4f}")
        logging.info(f"  Range: [{row['Min Accuracy']:.4f}, {row['Max Accuracy']:.4f}]")
        logging.info(f"  N: {int(row['N Seeds'])}")
        
    
    print("="*70)
    
    
    # Compute improvements over baseline
    series = summary_df[summary_df['Configuration'].str.contains('SGD baseline')]['Mean Accuracy']
    iloc_attr = getattr(series, 'iloc', None)
    baseline_acc = iloc_attr[0] if iloc_attr is not None else series[0]
    
    logging.info("IMPROVEMENT OVER BASELINE (SGD):")
    print("="*70)
    
    for idx, row in summary_df.iterrows():
        if 'baseline' in row['Configuration']:
            continue
        
        improvement = row['Mean Accuracy'] - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        
        logging.info(f"{row['Configuration']}")
        logging.info(f"  Δ Accuracy: +{improvement:.4f} ({improvement_pct:+.2f}%)")
    
    print("="*70)


def plot_ablation_results(summary_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot ablation study results as bar chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    configs = summary_df['Configuration'].values
    from src.utils.plot_helpers import arr_to_numpy_float
    means = arr_to_numpy_float(summary_df['Mean Accuracy'].values).astype(np.float64)
    stds = arr_to_numpy_float(summary_df['Std Accuracy'].values).astype(np.float64)
    
    # Color coding: baseline in gray, others by performance
    colors = []
    for config in configs:
        if 'baseline' in config.lower():
            colors.append('#808080')  # Gray
        elif 'adamw' in config.lower():
            colors.append('#2ecc71')  # Green (best)
        elif 'adam' in config.lower():
            colors.append('#3498db')  # Blue
        elif 'momentum' in config.lower():
            colors.append('#e74c3c')  # Red
        elif 'rmsprop' in config.lower():
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#95a5a6')  # Light gray
    
    # Bar plot
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=8, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Component-wise Contribution\n(Mean ± Std across seeds)', 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Annotate bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.005,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add horizontal line at baseline
    baseline_mean = means[0]  # Assuming first is baseline after sorting
    ax.axhline(y=baseline_mean, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (SGD)')
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        logging.info(f"Ablation plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def perform_statistical_comparisons(results: Dict[str, List[pd.DataFrame]]):
    """Perform pairwise statistical comparisons."""
    print("\n" + "="*70)
    logging.info("STATISTICAL COMPARISONS (T-TESTS)")
    print("="*70)
    
    # Extract final accuracies for each config
    config_accuracies = {}
    for config_name, dfs in results.items():
        accs = []
        for df in dfs:
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                series = eval_df['test_accuracy']
                iloc_attr = getattr(series, 'iloc', None)
                accs.append(iloc_attr[-1] if iloc_attr is not None else series[-1])
        config_accuracies[config_name] = np.array(accs)
    
    # Compare each configuration against baseline
    baseline_name = '1_SGD_baseline'
    baseline_accs = config_accuracies[baseline_name]
    
    for config_name, accs in config_accuracies.items():
        if config_name == baseline_name:
            continue
        
        result = compare_optimizers_ttest(
            accs, baseline_accs,
            name_A=config_name.replace('_', ' '),
            name_B='SGD Baseline',
            metric='test_accuracy'
        )
        
        print_ttest_results(result)


def main():
    """Run full ablation study."""
    
    # Base configuration
    base_config = {
        'dataset': 'MNIST',
        'model': 'SimpleMLP',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 10,
        'batch_size': 128
    }
    
    # Seeds for reproducibility
    seeds = [1, 2, 3, 4, 5]
    
    # Run ablation study
    results = run_ablation_study(base_config, seeds)
    
    # Analyze results
    summary_df = analyze_ablation_results(results)
    
    # Print summary
    print_ablation_summary(summary_df)
    
    # Save summary
    os.makedirs('results/ablation', exist_ok=True)
    summary_df.to_csv('results/ablation/ablation_summary.csv', index=False)
    logging.info("Summary saved to: results/ablation/ablation_summary.csv\n")
    
    # Plot results
    os.makedirs('plots', exist_ok=True)
    plot_ablation_results(summary_df, save_path='plots/ablation_study.png')
    
    # Statistical comparisons
    perform_statistical_comparisons(results)
    
    print("\n" + "="*70)
    logging.info("ABLATION STUDY COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
