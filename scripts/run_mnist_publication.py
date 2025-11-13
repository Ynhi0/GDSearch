#!/usr/bin/env python3
"""
Run publication-quality MNIST experiments with multiple seeds.

This script runs comprehensive MNIST experiments for scientific publication:
- 10 seeds per optimizer for robust statistics
- 5 optimizers: SGD, SGD+Momentum, Adam, AdamW, AMSGrad
- Statistical analysis with paired tests and Holm-Bonferroni correction
- Effect sizes and power analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from tqdm import tqdm
import pandas as pd
from src.experiments.run_nn_experiment import train_and_evaluate
from src.experiments.run_full_analysis import main as run_statistical_analysis


def run_mnist_experiments(seeds=None, results_dir='results'):
    """
    Run MNIST experiments with multiple seeds.
    
    Args:
        seeds: List of random seeds (default: [1-10])
        results_dir: Directory to save results
    """
    if seeds is None:
        seeds = list(range(1, 11))  # 10 seeds for publication
    
    print("=" * 80)
    print("MNIST EXPERIMENTS - PUBLICATION QUALITY")
    print("=" * 80)
    print(f"Number of seeds: {len(seeds)}")
    print(f"Seeds: {seeds}")
    print("=" * 80)
    
    # Define optimizer configurations
    optimizer_configs = [
        {
            'name': 'SGD',
            'optimizer': 'SGD',
            'lr': 0.01,
            'tag': 'publication'
        },
        {
            'name': 'SGD_Momentum',
            'optimizer': 'SGD_Momentum',
            'lr': 0.05,
            'momentum': 0.9,
            'tag': 'publication'
        },
        {
            'name': 'Adam',
            'optimizer': 'Adam',
            'lr': 0.001,
            'tag': 'publication'
        },
        {
            'name': 'AdamW',
            'optimizer': 'AdamW',
            'lr': 0.001,
            'weight_decay': 0.0001,
            'tag': 'publication'
        },
        {
            'name': 'AMSGrad',
            'optimizer': 'AMSGrad',
            'lr': 0.001,
            'tag': 'publication'
        }
    ]
    
    # Generate all experiment configs
    all_configs = []
    for opt_config in optimizer_configs:
        for seed in seeds:
            config = {
                'model': 'SimpleMLP',
                'dataset': 'MNIST',
                'optimizer': opt_config['optimizer'],
                'lr': opt_config['lr'],
                'epochs': 10,
                'batch_size': 128,
                'seed': seed,
                'tag': opt_config['tag'],
                'results_dir': results_dir
            }
            # Add optimizer-specific params
            if 'momentum' in opt_config:
                config['momentum'] = opt_config['momentum']
            if 'weight_decay' in opt_config:
                config['weight_decay'] = opt_config['weight_decay']
            
            all_configs.append(config)
    
    print(f"\nTotal experiments to run: {len(all_configs)}")
    print(f"\nRunning experiments...")
    
    # Run all experiments with progress bar
    results = []
    for config in tqdm(all_configs, desc="MNIST experiments"):
        try:
            df = train_and_evaluate(config)
            results.append(df)
        except Exception as e:
            print(f"\n❌ Error in experiment {config['optimizer']} seed {config['seed']}: {e}")
            continue
    
    print(f"\n✅ Completed {len(results)}/{len(all_configs)} experiments")
    
    # Verify all files were created
    print("\n" + "=" * 80)
    print("VERIFYING RESULTS")
    print("=" * 80)
    
    expected_files = []
    for opt_config in optimizer_configs:
        for seed in seeds:
            opt_name = opt_config['name']
            lr = opt_config['lr']
            filename = f"NN_SimpleMLP_MNIST_{opt_name}_lr{lr}_seed{seed}_publication.csv"
            expected_files.append(filename)
    
    found_files = []
    missing_files = []
    for fname in expected_files:
        fpath = Path(results_dir) / fname
        if fpath.exists():
            found_files.append(fname)
        else:
            missing_files.append(fname)
    
    print(f"✅ Found: {len(found_files)}/{len(expected_files)} result files")
    if missing_files:
        print(f"\n⚠️  Missing files:")
        for fname in missing_files[:5]:
            print(f"   - {fname}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
    
    return results


def run_statistical_comparison(results_dir='results'):
    """
    Run statistical analysis comparing all optimizers.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Define optimizer pairs to compare
    comparisons = [
        ('Adam', 'SGD'),
        ('AdamW', 'Adam'),
        ('AMSGrad', 'Adam'),
        ('SGD_Momentum', 'SGD'),
        ('AdamW', 'SGD'),
        ('AMSGrad', 'SGD'),
        ('AMSGrad', 'AdamW'),
        ('SGD_Momentum', 'Adam'),
    ]
    
    all_results = []
    
    for opt_a, opt_b in comparisons:
        print(f"\n📊 Comparing {opt_a} vs {opt_b}...")
        
        try:
            # Find result files
            import glob
            pattern_a = f"{results_dir}/NN_SimpleMLP_MNIST_{opt_a}_*_publication.csv"
            pattern_b = f"{results_dir}/NN_SimpleMLP_MNIST_{opt_b}_*_publication.csv"
            
            files_a = glob.glob(pattern_a)
            files_b = glob.glob(pattern_b)
            
            if not files_a or not files_b:
                print(f"   ⚠️  Missing result files")
                continue
            
            # Load results and extract final metrics
            import re
            
            def extract_seed(fname):
                match = re.search(r'seed(\d+)', fname)
                return int(match.group(1)) if match else None
            
            # Build seed-matched datasets
            data_a = {}
            for f in files_a:
                seed = extract_seed(f)
                if seed:
                    df = pd.read_csv(f)
                    final_row = df.iloc[-1]
                    data_a[seed] = final_row['test_loss']
            
            data_b = {}
            for f in files_b:
                seed = extract_seed(f)
                if seed:
                    df = pd.read_csv(f)
                    final_row = df.iloc[-1]
                    data_b[seed] = final_row['test_loss']
            
            # Get common seeds
            common_seeds = sorted(set(data_a.keys()) & set(data_b.keys()))
            
            if len(common_seeds) < 3:
                print(f"   ⚠️  Insufficient common seeds: {len(common_seeds)}")
                continue
            
            # Extract matched values
            values_a = [data_a[s] for s in common_seeds]
            values_b = [data_b[s] for s in common_seeds]
            
            # Compute statistics
            import numpy as np
            from scipy import stats
            
            mean_a = np.mean(values_a)
            std_a = np.std(values_a, ddof=1)
            mean_b = np.mean(values_b)
            std_b = np.std(values_b, ddof=1)
            
            # Test for normality
            _, p_norm_a = stats.shapiro(values_a)
            _, p_norm_b = stats.shapiro(values_b)
            
            # Choose test based on normality
            if p_norm_a > 0.05 and p_norm_b > 0.05:
                # Paired t-test
                stat, pval = stats.ttest_rel(values_a, values_b)
                test_name = 'Paired t-test'
                
                # Cohen's d
                diff = np.array(values_a) - np.array(values_b)
                cohens_d = np.mean(diff) / np.std(diff, ddof=1)
            else:
                # Wilcoxon signed-rank test
                stat, pval = stats.wilcoxon(values_a, values_b)
                test_name = 'Wilcoxon'
                
                # Rank-biserial correlation
                n = len(values_a)
                cohens_d = 1 - (2 * stat) / (n * (n + 1))
            
            # Power analysis
            from src.analysis.statistical_analysis import power_analysis_report
            power_report = power_analysis_report(values_a, values_b, cohens_d)
            
            result = {
                'Optimizer A': opt_a,
                'Optimizer B': opt_b,
                'n': len(common_seeds),
                'n_common_seeds': len(common_seeds),
                'Mean A': mean_a,
                'Std A': std_a,
                'Mean B': mean_b,
                'Std B': std_b,
                'Test': test_name,
                'p-value': pval,
                'Significant (α=0.05)': pval < 0.05,
                "Cohen's d": cohens_d,
                'Observed power': power_report['observed_power'],
                'Required n (80%)': power_report['required_n_80']
            }
            
            all_results.append(result)
            
            print(f"   {test_name}: p={pval:.4f}, d={cohens_d:.3f}, power={power_report['observed_power']:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("\n⚠️  No statistical comparisons could be computed")
        return
    
    # Apply Holm-Bonferroni correction
    from scipy.stats import false_discovery_control
    pvalues = df['p-value'].values
    
    # Holm-Bonferroni: sort p-values, compare to α/(m+1-k)
    n_tests = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    corrected_sig = np.zeros(n_tests, dtype=bool)
    
    for k, idx in enumerate(sorted_indices):
        alpha_corrected = 0.05 / (n_tests - k)
        if pvalues[idx] < alpha_corrected:
            corrected_sig[idx] = True
        else:
            break  # All remaining p-values are not significant
    
    df['Significant (Holm-Bonferroni)'] = corrected_sig
    
    # Save results
    output_path = f"{results_dir}/mnist_statistical_comparisons_publication.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Statistical analysis saved to: {output_path}")
    print(f"\nSignificant differences (after Holm-Bonferroni correction): {corrected_sig.sum()}/{n_tests}")
    
    return df


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run publication-quality MNIST experiments')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5,6,7,8,9,10',
                        help='Comma-separated list of seeds (default: 1-10)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Skip experiments, only run statistical analysis')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    print("\n" + "=" * 80)
    print(" MNIST PUBLICATION EXPERIMENTS ")
    print("=" * 80)
    print(f"\nSeeds: {seeds}")
    print(f"Results directory: {args.results_dir}")
    print(f"Total experiments: {len(seeds) * 5} (5 optimizers × {len(seeds)} seeds)")
    print("\n" + "=" * 80)
    
    if not args.stats_only:
        # Run experiments
        run_mnist_experiments(seeds=seeds, results_dir=args.results_dir)
    
    # Run statistical analysis
    run_statistical_comparison(results_dir=args.results_dir)
    
    print("\n" + "=" * 80)
    print("✅ MNIST PUBLICATION EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python scripts/generate_latex_tables.py")
    print("   → Generates LaTeX tables and Excel files")
    print("2. Check: results/mnist_statistical_comparisons_publication.csv")
    print("   → Statistical comparison results")
    print("3. Review: results/RESULTS_README.md")
    print("   → Guide for using results in your paper")
    print("=" * 80)


if __name__ == '__main__':
    main()
