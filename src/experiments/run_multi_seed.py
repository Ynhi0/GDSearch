"""
Multi-seed experiment runner for statistical analysis.
"""

import os
import sys
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from run_nn_experiment import train_and_evaluate, result_filename


def run_multi_seed_experiment(base_config: Dict[str, Any], seeds: List[int], results_dir: str = 'results') -> List[str]:
    """
    Run the same experiment with multiple seeds.
    
    Args:
        base_config: Base configuration dictionary
        seeds: List of random seeds
        results_dir: Directory to save results
        
    Returns:
        List of result file paths
    """
    os.makedirs(results_dir, exist_ok=True)
    result_files = []
    
    print(f"\n{'='*60}")
    print(f"Running Multi-Seed Experiment")
    print(f"Seeds: {seeds}")
    print(f"Base config: {base_config['model']}/{base_config['dataset']}/{base_config['optimizer']}")
    print(f"{'='*60}\n")
    
    for seed in tqdm(seeds, desc="Seeds"):
        config = base_config.copy()
        config['seed'] = seed
        config['tag'] = f'seed{seed}'
        
        # Run experiment
        df = train_and_evaluate(config)
        
        # Save result
        filename = result_filename(config)
        filepath = os.path.join(results_dir, filename)
        df.to_csv(filepath, index=False)
        result_files.append(filepath)
        
        print(f"  Seed {seed}: {filepath}")
    
    return result_files


def aggregate_results(result_files: List[str], metric: str = 'test_accuracy') -> Dict[str, Any]:
    """
    Aggregate results from multiple seeds.
    
    Args:
        result_files: List of CSV file paths
        metric: Metric to aggregate ('test_accuracy', 'test_loss', etc.)
        
    Returns:
        Dictionary with mean, std, min, max, values
    """
    values = []
    
    for filepath in result_files:
        df = pd.read_csv(filepath)
        eval_df = df[df['phase'] == 'eval']
        if not eval_df.empty:
            final_value = eval_df[metric].iloc[-1]
            values.append(final_value)
    
    values = np.array(values)
    
    return {
        'mean': values.mean(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'values': values.tolist(),
        'n': len(values)
    }


def print_aggregated_results(results: Dict[str, Any], metric_name: str = "Test Accuracy"):
    """Print aggregated results in a nice format."""
    print(f"\n{'='*60}")
    print(f"Aggregated Results: {metric_name}")
    print(f"{'='*60}")
    print(f"Mean:     {results['mean']:.4f}")
    print(f"Std:      {results['std']:.4f}")
    print(f"Min:      {results['min']:.4f}")
    print(f"Max:      {results['max']:.4f}")
    print(f"Range:    {results['max'] - results['min']:.4f}")
    print(f"N:        {results['n']}")
    print(f"\nFormatted: {results['mean']:.4f} ± {results['std']:.4f} (n={results['n']})")
    print(f"{'='*60}\n")


def save_aggregated_results(results: Dict[str, Any], filepath: str):
    """Save aggregated results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Aggregated results saved to: {filepath}")


def main():
    """Example usage."""
    # Example: Run AdamW on MNIST with 5 seeds
    base_config = {
        'model': 'SimpleMLP',
        'dataset': 'MNIST',
        'optimizer': 'AdamW',
        'lr': 0.001,
        'weight_decay': 0.0,
        'epochs': 5,
        'batch_size': 128,
    }
    
    seeds = [1, 2, 3, 4, 5]
    
    # Run experiments
    result_files = run_multi_seed_experiment(base_config, seeds)
    
    # Aggregate test accuracy
    acc_results = aggregate_results(result_files, metric='test_accuracy')
    print_aggregated_results(acc_results, metric_name="Test Accuracy")
    save_aggregated_results(acc_results, 'results/multiseed_test_accuracy.json')
    
    # Aggregate test loss
    loss_results = aggregate_results(result_files, metric='test_loss')
    print_aggregated_results(loss_results, metric_name="Test Loss")
    save_aggregated_results(loss_results, 'results/multiseed_test_loss.json')


if __name__ == "__main__":
    main()
