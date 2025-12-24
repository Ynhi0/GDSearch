#!/usr/bin/env python3
"""Extract best validation performance from partial results for publication."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
parser = argparse.ArgumentParser(description='Extract best-validation-epoch performance from partial results')
parser.add_argument('--results-dir', default=os.environ.get('GDSEARCH_RESULTS_DIR', 'results/results_full'), help='Path to results directory')
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
MNIST_DIR = RESULTS_DIR / "experiments" / "mnist" / "experiments" / "mnist"
CIFAR10_DIR = RESULTS_DIR / "experiments" / "cifar10"

logging.info(f"Using results directory: {RESULTS_DIR}")

def extract_best_performance(csv_path):
    """Extract best validation accuracy and corresponding test accuracy."""
    df = pd.read_csv(csv_path)
    
    # Find epoch with best validation accuracy
    if 'val_acc' in df.columns:
        best_idx = df['val_acc'].idxmax()
        best_val_acc = df.loc[best_idx, 'val_acc']
        test_acc_at_best_val = df.loc[best_idx, 'test_acc']
        best_epoch = df.loc[best_idx, 'epoch']
        final_epoch = len(df)
        
        return {
            'best_val_acc': best_val_acc,
            'test_acc_at_best_val': test_acc_at_best_val,
            'best_epoch': best_epoch,
            'final_epoch': final_epoch,
            'final_test_acc': df.iloc[-1]['test_acc']
        }
    return None

def main():
    print("=" * 80)
    print("EXTRACTING BEST PERFORMANCE FROM PARTIAL RESULTS")
    print("=" * 80)
    
    # Process MNIST
    mnist_results = []
    
    for csv_file in sorted(MNIST_DIR.glob("*.csv")):
        # Parse filename: MNIST_SimpleMLP_Optimizer_seedSEED.csv
        parts = csv_file.stem.split("_")
        seed_part = [p for p in parts if p.startswith("seed")]
        if not seed_part:
            continue
            
        seed = int(seed_part[0].replace("seed", ""))
        optimizer = "_".join(parts[2:-1])  # Everything between SimpleMLP and seedXXX
        
        perf = extract_best_performance(csv_file)
        if perf:
            mnist_results.append({
                'optimizer': optimizer,
                'seed': seed,
                **perf
            })
    
    df_mnist = pd.DataFrame(mnist_results)
    
    # Compute statistics per optimizer
    print("\n" + "=" * 80)
    print("MNIST RESULTS - TEST ACCURACY AT BEST VALIDATION EPOCH")
    print("=" * 80)
    print(f"\n{'Optimizer':<20} {'Mean±Std':<15} {'Min':<8} {'Max':<8} {'Avg Epoch':<10}")
    print("-" * 80)
    
    for optimizer in sorted(df_mnist['optimizer'].unique()):
        subset = df_mnist[df_mnist['optimizer'] == optimizer]
        mean_acc = subset['test_acc_at_best_val'].mean()
        std_acc = subset['test_acc_at_best_val'].std()
        min_acc = subset['test_acc_at_best_val'].min()
        max_acc = subset['test_acc_at_best_val'].max()
        avg_epoch = subset['best_epoch'].mean()
        
        print(f"{optimizer:<20} {mean_acc:.2f}±{std_acc:.2f}   {min_acc:.2f}   {max_acc:.2f}   {avg_epoch:.1f}")
    
    # Save detailed results
    output_csv = RESULTS_DIR / "mnist_best_performance.csv"
    df_mnist.to_csv(output_csv, index=False)
    print(f"\n✅ Saved detailed results to: {output_csv}")
    
    # Process CIFAR10
    cifar10_results = []
    
    for csv_file in sorted(CIFAR10_DIR.glob("CIFAR10_*.csv")):
        parts = csv_file.stem.split("_")
        seed_part = [p for p in parts if p.startswith("seed")]
        if not seed_part:
            continue
            
        seed = int(seed_part[0].replace("seed", ""))
        optimizer = parts[2]  # ResNet18_Optimizer_seedXXX
        
        perf = extract_best_performance(csv_file)
        if perf:
            cifar10_results.append({
                'optimizer': optimizer,
                'seed': seed,
                **perf
            })
    
    if cifar10_results:
        df_cifar10 = pd.DataFrame(cifar10_results)
        
        print("\n" + "=" * 80)
        print("CIFAR10 RESULTS - TEST ACCURACY AT BEST VALIDATION EPOCH")
        print("=" * 80)
        print(f"\n{'Optimizer':<20} {'Mean±Std':<15} {'Min':<8} {'Max':<8} {'Avg Epoch':<10}")
        print("-" * 80)
        
        for optimizer in sorted(df_cifar10['optimizer'].unique()):
            subset = df_cifar10[df_cifar10['optimizer'] == optimizer]
            mean_acc = subset['test_acc_at_best_val'].mean()
            std_acc = subset['test_acc_at_best_val'].std()
            min_acc = subset['test_acc_at_best_val'].min()
            max_acc = subset['test_acc_at_best_val'].max()
            avg_epoch = subset['best_epoch'].mean()
            
            print(f"{optimizer:<20} {mean_acc:.2f}±{std_acc:.2f}   {min_acc:.2f}   {max_acc:.2f}   {avg_epoch:.1f}")
        
        output_csv = RESULTS_DIR / "cifar10_best_performance.csv"
        df_cifar10.to_csv(output_csv, index=False)
        print(f"\n✅ Saved detailed results to: {output_csv}")
    
    print("\n" + "=" * 80)
    print("PUBLICATION-READY ANALYSIS")
    print("=" * 80)
    print("\nMethodology statement for your paper:")
    print("-" * 80)
    print("""
Note: This script extracts best-validation epoch results to salvage runs that stopped early (for example, due to early stopping).
If full runs without early stopping are available, prefer final-epoch metrics for publication and rerun experiments when possible.
When using best-epoch extraction, document the methodology clearly in the Methods section of your manuscript.
""")
    
    print("\n✅ Extraction complete. If you used best-epoch methodology, document it; rerun without early stopping when possible for final publication-ready data.")

if __name__ == "__main__":
    main()
