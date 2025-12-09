#!/usr/bin/env python3
"""
Learning Rate Ablation Study

Systematically tests different learning rates across optimizers to:
1. Find optimal learning rate for each optimizer
2. Understand learning rate sensitivity
3. Compare optimizer robustness to learning rate choice
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.statistical_analysis import compare_two_optimizers


def create_learning_rate_configs(
    base_config: Dict,
    learning_rates: List[float],
    optimizers: List[str]
) -> List[Dict]:
    """
    Create experiment configurations for learning rate ablation.
    
    Args:
        base_config: Base configuration with model, dataset, etc.
        learning_rates: List of learning rates to test
        optimizers: List of optimizer names
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for optimizer in optimizers:
        for lr in learning_rates:
            config = base_config.copy()
            config.update({
                'optimizer': optimizer,
                'lr': lr,
                'name': f"{optimizer}_lr{lr:.1e}"
            })
            configs.append(config)
    
    return configs


def run_learning_rate_ablation(
    base_config: Dict,
    learning_rates: List[float] = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    optimizers: List[str] = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW'],
    seeds: List[int] = [1, 2, 3, 4, 5],
    results_dir: str = 'results/lr_ablation'
) -> Dict[str, pd.DataFrame]:
    """
    Run learning rate ablation study with multiple seeds.
    
    Args:
        base_config: Base experiment configuration
        learning_rates: List of learning rates to test
        optimizers: List of optimizers to compare
        seeds: Random seeds for reproducibility
        results_dir: Output directory
        
    Returns:
        Dictionary mapping config names to aggregated results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*80)
    print("LEARNING RATE ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {base_config.get('dataset', 'MNIST')}")
    print(f"Model: {base_config.get('model', 'SimpleMLP')}")
    print(f"Learning rates: {learning_rates}")
    print(f"Optimizers: {optimizers}")
    print(f"Seeds: {seeds}")
    print("="*80)
    
    # Create configurations
    configs = create_learning_rate_configs(base_config, learning_rates, optimizers)
    
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
            config_with_seed = config.copy()
            config_with_seed['seed'] = seed
            
            # Import and run experiment
            try:
                from src.experiments.run_nn_experiment import train_and_evaluate
                df = train_and_evaluate(config_with_seed)
                
                # Save individual result
                filename = f"{config_name}_seed{seed}.csv"
                filepath = os.path.join(results_dir, filename)
                df.to_csv(filepath, index=False)
                
                seed_results.append(df)
                
                # Print final accuracy
                eval_df = df[df['phase'] == 'eval']
                if not eval_df.empty:
                    final_acc = eval_df['test_accuracy'].iloc[-1]
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
    print("Learning rate ablation completed!")
    print("="*80)
    
    return results


def analyze_learning_rate_results(
    results: Dict[str, pd.DataFrame],
    learning_rates: List[float],
    optimizers: List[str]
) -> pd.DataFrame:
    """
    Analyze learning rate ablation results.
    
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for optimizer in optimizers:
        for lr in learning_rates:
            config_name = f"{optimizer}_lr{lr:.1e}"
            
            if config_name not in results:
                continue
            
            df = results[config_name]
            
            # Extract final test accuracies from all seeds
            eval_df = df[df['phase'] == 'eval']
            
            if eval_df.empty:
                continue
            
            # Group by seed and get final accuracy
            final_accs = []
            for seed in eval_df['seed'].unique():
                seed_df = eval_df[eval_df['seed'] == seed]
                if not seed_df.empty:
                    final_accs.append(seed_df['test_accuracy'].iloc[-1])
            
            if final_accs:
                summary_data.append({
                    'Optimizer': optimizer,
                    'Learning Rate': lr,
                    'Mean Accuracy': np.mean(final_accs),
                    'Std Accuracy': np.std(final_accs),
                    'Min Accuracy': np.min(final_accs),
                    'Max Accuracy': np.max(final_accs),
                    'N Seeds': len(final_accs)
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    return summary_df


def plot_learning_rate_trends(
    summary_df: pd.DataFrame,
    save_path: str = None
):
    """
    Plot learning rate vs accuracy trends for all optimizers.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    optimizers = summary_df['Optimizer'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))
    
    for i, optimizer in enumerate(optimizers):
        opt_df = summary_df[summary_df['Optimizer'] == optimizer]
        opt_df = opt_df.sort_values('Learning Rate')
        
        lrs = opt_df['Learning Rate'].values
        means = opt_df['Mean Accuracy'].values
        stds = opt_df['Std Accuracy'].values
        
        # Plot line with error bars
        ax.errorbar(lrs, means, yerr=stds, 
                   marker='o', markersize=8, linewidth=2.5,
                   capsize=5, capthick=2, label=optimizer,
                   color=colors[i], alpha=0.8)
        
        # Mark optimal learning rate
        best_idx = opt_df['Mean Accuracy'].idxmax()
        best_lr = opt_df.loc[best_idx, 'Learning Rate']
        best_acc = opt_df.loc[best_idx, 'Mean Accuracy']
        ax.plot(best_lr, best_acc, 'r*', markersize=15, 
               markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (Mean ± Std)', fontsize=13, fontweight='bold')
    ax.set_title('Learning Rate Ablation: Impact on Optimizer Performance', 
                fontsize=15, fontweight='bold', pad=15)
    
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate trend plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_learning_rate_heatmap(
    summary_df: pd.DataFrame,
    save_path: str = None
):
    """
    Plot heatmap of accuracy across optimizers and learning rates.
    """
    # Pivot data for heatmap
    pivot_df = summary_df.pivot(
        index='Optimizer',
        columns='Learning Rate',
        values='Mean Accuracy'
    )
    
    # Format column labels
    pivot_df.columns = [f'{lr:.1e}' for lr in pivot_df.columns]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='RdYlGn',
               linewidths=0.5, cbar_kws={'label': 'Mean Test Accuracy'},
               ax=ax, vmin=pivot_df.min().min(), vmax=pivot_df.max().max())
    
    ax.set_title('Learning Rate Ablation Heatmap', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('Optimizer', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate heatmap saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def print_learning_rate_summary(summary_df: pd.DataFrame):
    """Print formatted learning rate ablation summary."""
    print("\n" + "="*80)
    print("LEARNING RATE ABLATION RESULTS")
    print("="*80)
    
    optimizers = summary_df['Optimizer'].unique()
    
    for optimizer in optimizers:
        print(f"\n{optimizer}:")
        print("-" * 80)
        opt_df = summary_df[summary_df['Optimizer'] == optimizer]
        opt_df = opt_df.sort_values('Learning Rate')
        
        for _, row in opt_df.iterrows():
            print(f"  LR {row['Learning Rate']:.1e}: "
                  f"{row['Mean Accuracy']:.4f} ± {row['Std Accuracy']:.4f} "
                  f"(n={int(row['N Seeds'])})")
        
        # Find optimal learning rate
        best_idx = opt_df['Mean Accuracy'].idxmax()
        best_lr = opt_df.loc[best_idx, 'Learning Rate']
        best_acc = opt_df.loc[best_idx, 'Mean Accuracy']
        print(f"  → Optimal: LR {best_lr:.1e} ({best_acc:.4f})")
    
    print("\n" + "="*80)


def main():
    """Run full learning rate ablation study."""
    
    # Base configuration
    base_config = {
        'dataset': 'MNIST',
        'model': 'SimpleMLP',
        'weight_decay': 0.0,
        'epochs': 10,
        'batch_size': 128
    }
    
    # Test configurations
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    optimizers = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW']
    seeds = [1, 2, 3, 4, 5]
    
    # Run ablation study
    results = run_learning_rate_ablation(
        base_config,
        learning_rates=learning_rates,
        optimizers=optimizers,
        seeds=seeds
    )
    
    # Analyze results
    summary_df = analyze_learning_rate_results(results, learning_rates, optimizers)
    
    # Print summary
    print_learning_rate_summary(summary_df)
    
    # Save summary
    os.makedirs('results/lr_ablation', exist_ok=True)
    summary_df.to_csv('results/lr_ablation/learning_rate_summary.csv', index=False)
    print("\nSummary saved to: results/lr_ablation/learning_rate_summary.csv")
    
    # Create visualizations
    os.makedirs('plots', exist_ok=True)
    plot_learning_rate_trends(summary_df, save_path='plots/learning_rate_trends.png')
    plot_learning_rate_heatmap(summary_df, save_path='plots/learning_rate_heatmap.png')
    
    print("\n" + "="*80)
    print("LEARNING RATE ABLATION STUDY COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
