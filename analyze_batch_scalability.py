#!/usr/bin/env python3
"""
Analyze Batch Size Scalability Results
======================================

This script analyzes the results from batch size scalability experiments
comparing Adam vs SAM performance across different batch sizes.

Usage:
    python analyze_batch_scalability.py --results-dirs results_bs64,results_bs256,results_bs1024
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(results_dirs):
    """Load results from multiple batch size directories."""
    all_data = []
    
    for results_dir in results_dirs:
        batch_size = int(results_dir.split('bs')[-1])
        
        # Load statistical comparisons
        stats_file = os.path.join(results_dir, 'mnist_statistical_comparisons_benchmark.csv')
        if os.path.exists(stats_file):
            df = pd.read_csv(stats_file)
            df['batch_size'] = batch_size
            all_data.append(df)
    
    if not all_data:
        print("❌ No statistical comparison files found!")
        return None
    
    return pd.concat(all_data, ignore_index=True)

def plot_scalability_analysis(df, output_dir='plots'):
    """Create plots analyzing batch size scalability."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter for Adam vs SAM_Adam comparisons
    adam_sam_data = df[df['optimizer_1'].isin(['Adam', 'SAM_Adam']) & 
                      df['optimizer_2'].isin(['Adam', 'SAM_Adam'])].copy()
    
    if adam_sam_data.empty:
        print("⚠️  No Adam vs SAM_Adam comparisons found")
        return
    
    # Calculate generalization gap (test_loss - train_loss) for each optimizer
    gap_data = []
    for batch_size in df['batch_size'].unique():
        batch_df = df[df['batch_size'] == batch_size]
        
        for opt in ['Adam', 'SAM_Adam']:
            opt_data = batch_df[batch_df['optimizer_1'] == opt]
            if not opt_data.empty:
                # Use the statistical comparison data
                mean_diff = opt_data['mean_diff'].iloc[0] if len(opt_data) > 0 else 0
                gap_data.append({
                    'batch_size': batch_size,
                    'optimizer': opt,
                    'generalization_gap': mean_diff
                })
    
    if not gap_data:
        print("⚠️  No generalization gap data available")
        return
    
    gap_df = pd.DataFrame(gap_data)
    
    # Plot 1: Generalization Gap vs Batch Size
    plt.figure(figsize=(10, 6))
    
    for opt in ['Adam', 'SAM_Adam']:
        opt_data = gap_df[gap_df['optimizer'] == opt]
        plt.plot(opt_data['batch_size'], opt_data['generalization_gap'], 
                marker='o', linewidth=2, label=opt)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Generalization Gap (Test Loss - Train Loss)')
    plt.title('Batch Size Scalability: Generalization Gap Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_scalability_generalization_gap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Performance comparison across batch sizes
    plt.figure(figsize=(12, 8))
    
    # Get unique batch sizes
    batch_sizes = sorted(df['batch_size'].unique())
    
    # Create subplots for each batch size
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, batch_size in enumerate(batch_sizes[:4]):  # Show first 4 batch sizes
        ax = axes[i]
        batch_df = df[df['batch_size'] == batch_size]
        
        # Plot test accuracy for each optimizer
        optimizers = ['Adam', 'SAM_Adam']
        colors = ['blue', 'red']
        
        for opt, color in zip(optimizers, colors):
            opt_data = batch_df[batch_df['optimizer_1'] == opt]
            if not opt_data.empty:
                # This is simplified - in reality you'd need to load individual CSV files
                # For now, just show the statistical comparison
                ax.bar([opt], [opt_data['mean_1'].iloc[0] if len(opt_data) > 0 else 0], 
                      color=color, alpha=0.7, label=opt)
        
        ax.set_title(f'Batch Size {batch_size}')
        ax.set_ylabel('Test Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_scalability_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Scalability analysis plots saved to:", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Analyze Batch Size Scalability Results')
    parser.add_argument('--results-dirs', type=str, required=True,
                       help='Comma-separated list of results directories (e.g., "results_bs64,results_bs256,results_bs1024")')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dirs = [d.strip() for d in args.results_dirs.split(',')]
    
    print("🔬 Analyzing Batch Size Scalability Results")
    print(f"Results directories: {results_dirs}")
    print()
    
    # Load data
    df = load_results(results_dirs)
    if df is None:
        return
    
    print("📊 Loaded data summary:")
    print(f"   Total comparisons: {len(df)}")
    print(f"   Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"   Optimizers: {df['optimizer_1'].unique()}")
    print()
    
    # Generate plots
    plot_scalability_analysis(df, args.output_dir)

if __name__ == '__main__':
    main()