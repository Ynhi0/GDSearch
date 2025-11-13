"""
Generate publication-ready experimental results with proper statistical rigor.

This script runs comprehensive experiments suitable for inclusion in a scientific paper:
1. Multi-seed MNIST experiments (10 seeds) with all optimizers
2. Statistical comparisons with effect sizes and power analysis
3. 2D optimization experiments on all test functions
4. Initial condition robustness analysis
5. Optimizer ablation study

All results are saved with proper formatting for tables and figures in academic papers.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.run_nn_experiment import train_and_evaluate, result_filename
from src.experiments.run_full_analysis import run_full_pipeline
from src.experiments.run_experiment import run_single_experiment, create_experiment_configs
from src.experiments.run_initial_condition_robustness import (
    run_robustness_experiment, generate_initial_points
)
from src.experiments.run_optimizer_ablation import run_optimizer_ablation
from src.core.test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint


def run_mnist_experiments(seeds: list = None, results_dir: str = 'results'):
    """
    Run comprehensive MNIST experiments with multiple optimizers and seeds.
    
    Args:
        seeds: List of random seeds (default: 10 seeds from 1-10)
        results_dir: Output directory
    """
    if seeds is None:
        seeds = list(range(1, 11))  # Seeds 1-10 for statistical validity
    
    print("="*80)
    print("MNIST EXPERIMENTS - PUBLICATION QUALITY")
    print("="*80)
    print(f"Number of seeds: {len(seeds)}")
    print(f"Seeds: {seeds}")
    print("="*80)
    
    # Define optimizer configurations for MNIST
    base_configs = [
        {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'SGD',
            'lr': 0.01,
            'epochs': 10,
            'batch_size': 128,
            'tag': 'final'
        },
        {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'SGD_Momentum',
            'lr': 0.05,
            'momentum': 0.9,
            'epochs': 10,
            'batch_size': 128,
            'tag': 'final'
        },
        {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'Adam',
            'lr': 0.001,
            'epochs': 10,
            'batch_size': 128,
            'tag': 'final'
        },
        {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'AdamW',
            'lr': 0.001,
            'weight_decay': 0.0001,
            'epochs': 10,
            'batch_size': 128,
            'tag': 'final'
        },
        {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'AMSGrad',
            'lr': 0.001,
            'epochs': 10,
            'batch_size': 128,
            'tag': 'final'
        }
    ]
    
    # Generate configs for all seeds
    all_configs = []
    for base_config in base_configs:
        for seed in seeds:
            config = base_config.copy()
            config['seed'] = seed
            all_configs.append(config)
    
    print(f"\nTotal experiments to run: {len(all_configs)}")
    print("\nRunning experiments...")
    
    from tqdm import tqdm
    results_files = []
    
    for config in tqdm(all_configs, desc="MNIST experiments"):
        try:
            df = train_and_evaluate(config)
            fname = result_filename(config)
            out_path = os.path.join(results_dir, fname)
            df.to_csv(out_path, index=False)
            results_files.append(out_path)
        except Exception as e:
            print(f"\n⚠️  Error in experiment {config.get('optimizer')}_seed{config.get('seed')}: {e}")
            continue
    
    print(f"\n✅ Completed {len(results_files)}/{len(all_configs)} experiments")
    return results_files


def run_statistical_analysis(results_dir: str = 'results', plots_dir: str = 'plots'):
    """
    Run comprehensive statistical analysis on MNIST results.
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    # Define comparison pairs (all pairwise comparisons)
    comparison_pairs = [
        ('SGD', 'SGD_Momentum'),
        ('SGD', 'Adam'),
        ('SGD', 'AdamW'),
        ('SGD', 'AMSGrad'),
        ('SGD_Momentum', 'Adam'),
        ('SGD_Momentum', 'AdamW'),
        ('SGD_Momentum', 'AMSGrad'),
        ('Adam', 'AdamW'),
        ('Adam', 'AMSGrad'),
        ('AdamW', 'AMSGrad'),
    ]
    
    # Create a minimal config for the analysis pipeline
    config = {
        'task': 'neural_network',
        'model': 'SimpleMLP',
        'dataset': 'MNIST',
    }
    
    # Save temp config
    config_path = os.path.join(results_dir, 'temp_analysis_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Run analysis (without re-running experiments)
    from src.analysis.statistical_analysis import (
        load_multiseed_results,
        extract_final_metric,
        auto_select_test,
        holm_bonferroni_correction,
        power_analysis_report
    )
    
    import glob
    
    summary_rows = []
    p_values = []
    
    for opt_A, opt_B in comparison_pairs:
        print(f"\nComparing {opt_A} vs {opt_B}...")
        
        pattern_A = f"*{opt_A}*seed*final.csv"
        pattern_B = f"*{opt_B}*seed*final.csv"
        
        files_A = glob.glob(os.path.join(results_dir, pattern_A))
        files_B = glob.glob(os.path.join(results_dir, pattern_B))
        
        if not files_A or not files_B:
            print(f"  ⚠️  Missing results, skipping")
            continue
        
        # Extract seeds and final accuracies
        def extract_seed_and_metric(fname):
            import re
            m = re.search(r"seed(\d+)", os.path.basename(fname))
            seed = int(m.group(1)) if m else None
            df = pd.read_csv(fname)
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                acc = float(eval_df['test_accuracy'].iloc[-1])
                return seed, acc
            return None, None
        
        map_A = {}
        map_B = {}
        
        for f in files_A:
            s, acc = extract_seed_and_metric(f)
            if s is not None and acc is not None:
                map_A[s] = acc
        
        for f in files_B:
            s, acc = extract_seed_and_metric(f)
            if s is not None and acc is not None:
                map_B[s] = acc
        
        common_seeds = sorted(set(map_A) & set(map_B))
        
        if len(common_seeds) < 3:
            print(f"  ⚠️  Insufficient common seeds ({len(common_seeds)}), skipping")
            continue
        
        results_A = np.array([map_A[s] for s in common_seeds])
        results_B = np.array([map_B[s] for s in common_seeds])
        
        # Statistical test
        auto = auto_select_test(results_A, results_B, paired=True, name_A=opt_A, name_B=opt_B)
        test_result = auto['test_result']
        test_type = auto.get('test_type', 'unknown')
        p_value = float(test_result.get('p_value', 1.0))
        p_values.append(p_value)
        
        # Power analysis
        power_report = power_analysis_report(results_A, results_B, name_A=opt_A, name_B=opt_B)
        
        # Build row
        row = {
            'Optimizer A': opt_A,
            'Optimizer B': opt_B,
            'n': len(common_seeds),
            'Mean A': results_A.mean(),
            'Std A': results_A.std(),
            'Mean B': results_B.mean(),
            'Std B': results_B.std(),
            'Test': test_type,
            'p-value': p_value,
            'Significant (α=0.05)': p_value < 0.05,
        }
        
        if 'cohens_d' in test_result:
            row['Cohen\'s d'] = test_result['cohens_d']
        if 'effect_size_r' in test_result:
            row['Effect size (r)'] = test_result['effect_size_r']
        
        row['Observed power'] = power_report['achieved_power']
        row['Required n (80%)'] = power_report['required_n']
        
        summary_rows.append(row)
        
        print(f"  Test: {test_type}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Power: {power_report['achieved_power']:.3f}")
    
    # Apply Holm-Bonferroni correction
    if p_values:
        corrected = holm_bonferroni_correction(p_values, alpha=0.05)
        for i, row in enumerate(summary_rows):
            row['Significant (Holm-Bonferroni)'] = corrected[i]
    
    # Save results
    df_stats = pd.DataFrame(summary_rows)
    stats_path = os.path.join(results_dir, 'mnist_statistical_comparisons_publication.csv')
    df_stats.to_csv(stats_path, index=False)
    
    print(f"\n✅ Statistical analysis saved to: {stats_path}")
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(df_stats.to_string(index=False))
    print("="*80)
    
    return df_stats


def run_2d_experiments(results_dir: str = 'results'):
    """
    Run comprehensive 2D optimization experiments.
    """
    print("\n" + "="*80)
    print("2D OPTIMIZATION EXPERIMENTS")
    print("="*80)
    
    # Run standard experiments
    configs = create_experiment_configs()
    
    print(f"Running {len(configs)} 2D experiments...")
    
    from tqdm import tqdm
    for config in tqdm(configs, desc="2D experiments"):
        df = run_single_experiment(
            optimizer_config=config['optimizer_config'],
            function_config=config['function_config'],
            initial_point=config['initial_point'],
            num_iterations=config['num_iterations'],
            seed=config['seed']
        )
        
        exp_id = config.get('experiment_id', 'unknown')
        filepath = os.path.join(results_dir, f"{exp_id}.csv")
        df.to_csv(filepath, index=False)
    
    print(f"✅ Completed {len(configs)} 2D experiments")


def run_robustness_analysis(results_dir: str = 'results', plots_dir: str = 'plots'):
    """
    Run initial condition robustness analysis.
    """
    print("\n" + "="*80)
    print("INITIAL CONDITION ROBUSTNESS ANALYSIS")
    print("="*80)
    
    # Test on all three test functions
    test_functions = [
        {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}, 'center': (-1.5, 2.0), 'radius': 2.5},
        {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}, 'center': (1.0, 1.0), 'radius': 2.0},
        {'type': 'SaddlePoint', 'params': {}, 'center': (0.5, 0.5), 'radius': 1.5},
    ]
    
    optimizer_configs = [
        {'type': 'SGD', 'params': {'lr': 0.001}},
        {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}},
        {'type': 'SGDNesterov', 'params': {'lr': 0.01, 'beta': 0.9}},
        {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9}},
        {'type': 'Adam', 'params': {'lr': 0.01}},
        {'type': 'AdamW', 'params': {'lr': 0.01, 'weight_decay': 0.01}},
        {'type': 'AMSGrad', 'params': {'lr': 0.01}},
    ]
    
    for func_cfg in test_functions:
        print(f"\nTesting on {func_cfg['type']}...")
        
        initial_points = generate_initial_points(
            center=func_cfg['center'],
            radius=func_cfg['radius'],
            num_points=30,  # 30 initial conditions for statistical validity
            seed=42
        )
        
        df_agg = run_robustness_experiment(
            optimizer_configs=optimizer_configs,
            function_config=func_cfg,
            initial_points=initial_points,
            max_iterations=5000,
            convergence_threshold=1e-6,
            results_dir=results_dir,
            plots_dir=plots_dir
        )
    
    print("✅ Robustness analysis complete")


def run_ablation_study(results_dir: str = 'results', plots_dir: str = 'plots'):
    """
    Run optimizer ablation study.
    """
    print("\n" + "="*80)
    print("OPTIMIZER ABLATION STUDY")
    print("="*80)
    
    rosenbrock = Rosenbrock(a=1, b=100)
    initial_point = (-1.5, 2.0)
    
    df_summary = run_optimizer_ablation(
        test_function=rosenbrock,
        initial_point=initial_point,
        max_iterations=10000,
        results_dir=results_dir,
        plots_dir=plots_dir
    )
    
    print("✅ Ablation study complete")


def main():
    """
    Run all publication-quality experiments.
    """
    print("\n" + "="*80)
    print(" PUBLICATION-QUALITY EXPERIMENT SUITE ")
    print("="*80)
    print("\nThis will generate comprehensive results for your scientific paper:")
    print("1. Multi-seed MNIST experiments (10 seeds × 5 optimizers = 50 runs)")
    print("2. Statistical analysis with Holm-Bonferroni correction")
    print("3. 2D optimization experiments on 3 test functions")
    print("4. Initial condition robustness analysis (30 initial points)")
    print("5. Optimizer ablation study")
    print("\nEstimated time: 15-20 minutes")
    print("="*80)
    
    response = input("\nProceed with full experiment suite? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    results_dir = 'results'
    plots_dir = 'plots'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. MNIST experiments
    print("\n" + "="*80)
    print("PHASE 1: MNIST NEURAL NETWORK EXPERIMENTS")
    print("="*80)
    seeds = list(range(1, 11))  # 10 seeds for statistical validity (n≥10 recommended)
    run_mnist_experiments(seeds=seeds, results_dir=results_dir)
    
    # 2. Statistical analysis
    print("\n" + "="*80)
    print("PHASE 2: STATISTICAL ANALYSIS")
    print("="*80)
    df_stats = run_statistical_analysis(results_dir=results_dir, plots_dir=plots_dir)
    
    # 3. 2D experiments
    print("\n" + "="*80)
    print("PHASE 3: 2D OPTIMIZATION EXPERIMENTS")
    print("="*80)
    run_2d_experiments(results_dir=results_dir)
    
    # 4. Robustness analysis
    print("\n" + "="*80)
    print("PHASE 4: ROBUSTNESS ANALYSIS")
    print("="*80)
    run_robustness_analysis(results_dir=results_dir, plots_dir=plots_dir)
    
    # 5. Ablation study
    print("\n" + "="*80)
    print("PHASE 5: ABLATION STUDY")
    print("="*80)
    run_ablation_study(results_dir=results_dir, plots_dir=plots_dir)
    
    # Final summary
    print("\n" + "="*80)
    print(" EXPERIMENT SUITE COMPLETE ")
    print("="*80)
    print("\n📊 Results ready for publication:")
    print(f"\n📁 Results directory: {results_dir}/")
    print("   - mnist_statistical_comparisons_publication.csv")
    print("   - optimizer_ablation_summary.csv")
    print("   - initial_condition_robustness_*.csv")
    print("   - Individual experiment CSVs")
    print(f"\n📈 Plots directory: {plots_dir}/")
    print("   - optimizer_ablation_study.png")
    print("   - initial_condition_robustness_*.png")
    print("   - statistical_*_vs_*.png")
    print("\n✅ All data is publication-ready with proper statistical rigor!")
    print("="*80)


if __name__ == '__main__':
    main()
