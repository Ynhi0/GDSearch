#!/usr/bin/env python3
"""
Learning Rate Scheduler Ablation Study

Tests different LR schedules across optimizers to:
1. Understand scheduler impact on convergence
2. Find optimal schedule for each optimizer
3. Compare scheduler-optimizer interactions
4. Analyze convergence speed vs final performance
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.statistical_analysis import compare_two_optimizers
from src.utils.num_utils import safe_to_float


def create_scheduler_configs(
    base_config: Dict,
    schedulers: List[str],
    optimizers: List[str]
) -> List[Dict]:
    """
    Create experiment configurations for scheduler ablation.
    
    Args:
        base_config: Base configuration with model, dataset, etc.
        schedulers: List of scheduler names
        optimizers: List of optimizer names
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    # Dynamic T_max to match epochs - prevents unwanted LR restart
    # If T_max < epochs, CosineAnnealing will restart and increase LR at the end
    epochs = base_config.get('epochs', 15)
    
    # GAP #15, #16, #17 FIX: Comprehensive scheduler parameter dictionary
    scheduler_params = {
        'None': {},
        'LinearWarmup': {'warmup_epochs': max(1, epochs // 10)},  # GAP #16: 10% warmup
        'StepLR_25': {'step_size': max(1, epochs // 4), 'gamma': 0.1},  # GAP #15: Decay at 25%
        'StepLR_33': {'step_size': max(1, epochs // 3), 'gamma': 0.1},  # GAP #15: Decay at 33%
        'StepLR_50': {'step_size': max(1, epochs // 2), 'gamma': 0.1},  # GAP #15: Decay at 50%
        'StepLR_75': {'step_size': max(1, 3 * epochs // 4), 'gamma': 0.1},  # GAP #15: Decay at 75%
        'StepLR': {'step_size': max(1, epochs // 3), 'gamma': 0.5},  # Keep original for backward compatibility
        'ExponentialLR': {'gamma': 0.95},
        'CosineAnnealingLR': {'T_max': epochs, 'eta_min': 1e-6},  # Match training length
        'OneCycleLR': {'max_lr': base_config.get('lr', 0.1) * 10, 'epochs': epochs,  # GAP #17
                       'steps_per_epoch': 1, 'pct_start': 0.3},
        'ReduceLROnPlateau': {'mode': 'min', 'factor': 0.5, 'patience': max(1, epochs // 10)}
    }
    
    for optimizer in optimizers:
        for scheduler in schedulers:
            config = base_config.copy()
            config.update({
                'optimizer': optimizer,
                'scheduler': scheduler,
                'scheduler_params': scheduler_params.get(scheduler, {}),
                'name': f"{optimizer}_{scheduler}"
            })
            configs.append(config)
    
    return configs


def run_scheduler_ablation(
    base_config: Dict,
    schedulers: List[str] = ['None', 'LinearWarmup', 'StepLR_25', 'StepLR_33', 'StepLR_50', 
                             'StepLR_75', 'ExponentialLR', 'CosineAnnealingLR', 'OneCycleLR'],
    optimizers: List[str] = ['SGD', 'Adam', 'AdamW'],
    seeds: List[int] = [1, 2, 3, 4, 5],
    results_dir: str = 'results/scheduler_ablation'
) -> Dict[str, pd.DataFrame]:
    """
    Run scheduler ablation study with multiple seeds.
    
    GAP #15, #16, #17 FIX: Comprehensive scheduler comparison
    - LinearWarmup (GAP #16): Essential for Adam + large batch training
    - Multiple StepLR configs (GAP #15): Fair comparison at 25%, 33%, 50%, 75% of training
    - OneCycleLR (GAP #17): Tests super-convergence phenomenon (Smith 2018)
    
    Args:
        base_config: Base experiment configuration
        schedulers: List of scheduler types to test
        optimizers: List of optimizers to compare
        seeds: Random seeds for reproducibility
        results_dir: Output directory
        
    Returns:
        Dictionary mapping config names to aggregated results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*80)
    print("LEARNING RATE SCHEDULER ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {base_config.get('dataset', 'MNIST')}")
    print(f"Model: {base_config.get('model', 'SimpleMLP')}")
    print(f"Schedulers: {schedulers}")
    print(f"Optimizers: {optimizers}")
    print(f"Seeds: {seeds}")
    print("="*80)
    
    # Create configurations
    configs = create_scheduler_configs(base_config, schedulers, optimizers)
    
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
                    final_acc = safe_to_float(ensure_series(eval_df['test_accuracy']).iloc[-1])
                    print(f"Acc: {final_acc:.4f}")
                else:
                    print("Done")
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Aggregate results
        if seed_results:
            results[config_name] = pd.concat(seed_results, ignore_index=True)
    
    print("\n" + "="*80)
    print("Scheduler ablation completed!")
    print("="*80)
    
    return results


def analyze_scheduler_results(
    results: Dict[str, pd.DataFrame],
    schedulers: List[str],
    optimizers: List[str]
) -> pd.DataFrame:
    """
    Analyze scheduler ablation results.
    
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for optimizer in optimizers:
        for scheduler in schedulers:
            config_name = f"{optimizer}_{scheduler}"
            
            if config_name not in results:
                continue
            
            from src.utils.type_guards import ensure_dataframe
            df = ensure_dataframe(results[config_name])
            
            # Extract final test accuracies from all seeds
            eval_df = ensure_dataframe(df[df['phase'] == 'eval'])
            
            if eval_df.empty:
                continue
            
            # Group by seed and get final accuracy
            final_accs = []
            convergence_epochs = []
            
            from src.utils.type_guards import ensure_series, ensure_dataframe
            from src.utils.plot_helpers import arr_to_numpy_float
            for seed in ensure_series(eval_df['seed']).unique():
                seed_df = ensure_dataframe(eval_df[eval_df['seed'] == seed])
                if seed_df.empty:
                    continue
                # Skip tainted seeds
                if 'tainted' in seed_df.columns:
                    tainted_series = ensure_series(seed_df['tainted'])
                    if bool(tainted_series.any()):
                        continue

                test_acc_series = ensure_series(seed_df['test_accuracy'])
                final_accs.append(safe_to_float(test_acc_series.iloc[-1]))

                # Estimate convergence epoch (when accuracy stabilizes)
                accs = arr_to_numpy_float(test_acc_series)
                if accs.size > 5:
                    diffs = np.abs(np.diff(accs))
                    converged_idx = np.where(diffs < 0.001)[0]
                    if converged_idx.size > 0:
                        convergence_epochs.append(int(converged_idx[0]))
            
            if final_accs:
                summary_data.append({
                    'Optimizer': optimizer,
                    'Scheduler': scheduler,
                    'Mean Accuracy': np.mean(final_accs),
                    'Std Accuracy': np.std(final_accs),
                    'Min Accuracy': np.min(final_accs),
                    'Max Accuracy': np.max(final_accs),
                    'Mean Convergence Epoch': np.mean(convergence_epochs) if convergence_epochs else np.nan,
                    'N Seeds': len(final_accs)
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    return summary_df


def plot_scheduler_comparison(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot scheduler comparison across optimizers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    optimizers = summary_df['Optimizer'].unique()
    schedulers = summary_df['Scheduler'].unique()
    
    # Plot 1: Accuracy by scheduler and optimizer
    x = np.arange(len(schedulers))
    width = 0.8 / len(optimizers)
    
    from matplotlib import cm
    cmap = cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(optimizers)))
    
    for i, optimizer in enumerate(optimizers):
        opt_df = summary_df[summary_df['Optimizer'] == optimizer]
        opt_df = opt_df.set_index('Scheduler').reindex(schedulers)
        
        means = opt_df['Mean Accuracy'].values
        stds = opt_df['Std Accuracy'].values
        
        ax1.bar(x + i * width, means, width, yerr=stds,
               label=optimizer, color=colors[i], alpha=0.8,
               capsize=5, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Scheduler', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (Mean ± Std)', fontsize=13, fontweight='bold')
    ax1.set_title('Scheduler Impact on Final Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(optimizers) - 1) / 2)
    ax1.set_xticklabels(schedulers, rotation=15, ha='right')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Convergence speed
    from typing import cast
    from src.utils.plot_helpers import arr_to_numpy_float
    for i, optimizer in enumerate(optimizers):
        opt_df = cast(pd.DataFrame, summary_df[summary_df['Optimizer'] == optimizer])
        opt_df = opt_df.set_index('Scheduler').reindex(schedulers)
        assert isinstance(opt_df, pd.DataFrame)
        
        conv_epochs = arr_to_numpy_float(opt_df['Mean Convergence Epoch'])
        
        ax2.plot(schedulers, conv_epochs, marker='o', markersize=10,
                linewidth=2.5, label=optimizer, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Scheduler', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Convergence Epoch', fontsize=13, fontweight='bold')
    ax2.set_title('Scheduler Impact on Convergence Speed', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which='both')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"Scheduler comparison plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def print_scheduler_summary(summary_df: pd.DataFrame):
    """Print formatted scheduler ablation summary."""
    from src.utils.type_guards import ensure_series, ensure_dataframe

    print("\n" + "="*80)
    print("LEARNING RATE SCHEDULER ABLATION RESULTS")
    print("="*80)
    
    optimizers = summary_df['Optimizer'].unique()
    
    for optimizer in optimizers:
        print(f"\n{optimizer}:")
        print("-" * 80)
        opt_df = ensure_dataframe(summary_df[summary_df['Optimizer'] == optimizer])
        
        for _, row in opt_df.iterrows():
            conv_val = row.get('Mean Convergence Epoch', np.nan)
            is_na = pd.isna(conv_val)
            if isinstance(is_na, (np.ndarray, pd.Series, pd.DataFrame)):
                is_na_bool = bool(is_na.any())
            else:
                is_na_bool = bool(is_na)
            conv_str = f"{conv_val:.1f}" if not is_na_bool else "N/A"
            print(f"  {row['Scheduler']:<20}: "
                  f"Acc={row['Mean Accuracy']:.4f}±{row['Std Accuracy']:.4f}, "
                  f"ConvEpoch={conv_str}")
        
        # Find best scheduler
        best_idx = ensure_series(opt_df['Mean Accuracy']).idxmax()
        best_scheduler = opt_df.loc[best_idx, 'Scheduler']
        best_acc = float(opt_df.loc[best_idx, 'Mean Accuracy'])


def main():
    """Run full scheduler ablation study."""
    
    # Base configuration
    base_config = {
        'dataset': 'MNIST',
        'model': 'SimpleMLP',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 15,
        'batch_size': 128
    }
    
    # Test configurations
    schedulers = ['None', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR']
    optimizers = ['SGD', 'Adam', 'AdamW']
    seeds = [1, 2, 3, 4, 5]
    
    # Run ablation study
    results = run_scheduler_ablation(
        base_config,
        schedulers=schedulers,
        optimizers=optimizers,
        seeds=seeds
    )
    
    # Analyze results
    summary_df = analyze_scheduler_results(results, schedulers, optimizers)
    
    # Print summary
    print_scheduler_summary(summary_df)
    
    # Save summary
    os.makedirs('results/scheduler_ablation', exist_ok=True)
    summary_df.to_csv('results/scheduler_ablation/scheduler_summary.csv', index=False)
    print("\nSummary saved to: results/scheduler_ablation/scheduler_summary.csv")
    
    # Create visualizations
    os.makedirs('plots', exist_ok=True)
    plot_scheduler_comparison(summary_df, save_path='plots/scheduler_comparison.png')
    
    print("\n" + "="*80)
    print("LEARNING RATE SCHEDULER ABLATION STUDY COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
