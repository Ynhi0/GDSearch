"""
Fair Optimizer Ablation Framework

Implements hyperparameter fairness protocols for scientifically rigorous optimizer comparisons.
Based on best practices from:
- Choi et al. "On Empirical Comparisons of Optimizers for Deep Learning" NeurIPS 2019
- Schmidt et al. "Descending through a Crowded Valley" ICML 2021

Key Features:
1. Equal tuning budget per optimizer
2. Optimizer-specific search spaces (from empirical research)
3. Automatic per-optimizer tuning or fair LR sweeps
4. Statistical significance testing with multiple comparison corrections
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


# Optimizer-specific hyperparameter ranges based on empirical research
# Sources: ImageNet training (Goyal CVPR 2017, He CVPR 2019), Adam paper (Kingma & Ba 2014)
OPTIMIZER_SEARCH_SPACES = {
    'SGD': {
        'lr': (0.001, 1.0, 'log'),  # SGD benefits from higher LR
        'momentum': (0.0, 0.99, 'uniform'),
    },
    'SGDMomentum': {
        'lr': (0.001, 0.5, 'log'),
        'momentum': (0.5, 0.99, 'uniform'),  # Momentum rarely optimal below 0.5
    },
    'SGDNesterov': {
        'lr': (0.001, 0.5, 'log'),
        'momentum': (0.5, 0.99, 'uniform'),
    },
    'RMSProp': {
        'lr': (0.0001, 0.01, 'log'),  # Adaptive methods prefer lower LR
        'alpha': (0.9, 0.999, 'uniform'),
        'epsilon': (1e-8, 1e-6, 'log'),
    },
    'Adam': {
        'lr': (0.0001, 0.01, 'log'),
        'beta1': (0.8, 0.95, 'uniform'),
        'beta2': (0.99, 0.9999, 'log'),  # Very close to 1.0
        'epsilon': (1e-8, 1e-7, 'log'),
    },
    'AdamW': {
        'lr': (0.0001, 0.01, 'log'),
        'weight_decay': (0.0, 0.1, 'log'),
        'beta1': (0.8, 0.95, 'uniform'),
        'beta2': (0.99, 0.9999, 'log'),
    },
    'AMSGrad': {
        'lr': (0.0001, 0.01, 'log'),
        'beta1': (0.8, 0.95, 'uniform'),
        'beta2': (0.99, 0.9999, 'log'),
    },
    'RAdam': {
        'lr': (0.0001, 0.01, 'log'),
        'beta1': (0.8, 0.95, 'uniform'),
        'beta2': (0.99, 0.9999, 'log'),
    },
    'AdaBound': {
        'lr': (0.0001, 0.01, 'log'),
        'final_lr': (0.01, 0.1, 'log'),
        'beta1': (0.8, 0.95, 'uniform'),
        'beta2': (0.99, 0.9999, 'log'),
    },
    'LAMB': {
        'lr': (0.0001, 0.01, 'log'),
        'weight_decay': (0.0, 0.1, 'log'),
        'beta1': (0.8, 0.95, 'uniform'),
        'beta2': (0.99, 0.9999, 'log'),
    },
}

# Published defaults from original papers (for baseline comparisons)
# NOTE: Parameter names may differ from constructor signatures (e.g., 'alpha' vs 'decay_rate')
# Use translate_optimizer_params() to convert to correct constructor kwargs
PUBLISHED_DEFAULTS = {
    'SGD': {
        'lr': 0.1,
        'momentum': 0.9,
        'source': 'Krizhevsky et al. ImageNet Classification 2012',
    },
    'SGDMomentum': {
        'lr': 0.01,
        'momentum': 0.9,
        'source': 'Standard SGD+Momentum baseline',
    },
    'Adam': {
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'source': 'Kingma & Ba Adam paper 2014',
    },
    'AdamW': {
        'lr': 0.001,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'source': 'Loshchilov & Hutter AdamW paper 2017',
    },
    'RMSProp': {
        'lr': 0.001,
        'alpha': 0.99,  # Maps to 'decay_rate' in src.core.optimizers.RMSProp
        'epsilon': 1e-8,
        'source': 'Hinton Coursera Lecture / TensorFlow defaults',
    },
}

# Parameter name translation map for optimizer constructor compatibility
PARAM_NAME_TRANSLATION = {
    'RMSProp': {
        'alpha': 'decay_rate',  # TensorFlow/PyTorch uses 'alpha', our implementation uses 'decay_rate'
    },
    # Add more translations as needed
}


def translate_optimizer_params(optimizer_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate parameter names from standard conventions to constructor-specific names.
    
    This handles cases where published defaults use different naming conventions
    than the actual optimizer implementation (e.g., 'alpha' vs 'decay_rate' for RMSProp).
    
    Args:
        optimizer_name: Name of the optimizer
        params: Dictionary of parameters with standard names
        
    Returns:
        Dictionary of parameters with constructor-compatible names
    """
    if optimizer_name not in PARAM_NAME_TRANSLATION:
        return params.copy()
    
    translated = params.copy()
    for old_name, new_name in PARAM_NAME_TRANSLATION[optimizer_name].items():
        if old_name in translated:
            translated[new_name] = translated.pop(old_name)
    
    return translated


def generate_lr_sweep(optimizer_name: str, n_points: int = 7) -> List[float]:
    """
    Generate learning rate sweep appropriate for optimizer type.
    
    Based on empirical optimal ranges:
    - SGD family: Higher LRs (0.001 - 1.0)
    - Adaptive methods: Lower LRs (0.0001 - 0.01)
    
    Args:
        optimizer_name: Name of optimizer
        n_points: Number of LR values to test
        
    Returns:
        List of learning rates in log-space
    """
    # Robust key matching: case-insensitive with normalization of separators
    matched_key = None
    
    # Normalize input: remove separators and convert to lowercase
    normalized_input = optimizer_name.replace('+', '').replace('-', '').replace('_', '').replace(' ', '').lower()
    
    # Find matching key in OPTIMIZER_SEARCH_SPACES
    for key in OPTIMIZER_SEARCH_SPACES.keys():
        normalized_key = key.replace('+', '').replace('-', '').replace('_', '').replace(' ', '').lower()
        if normalized_key == normalized_input:
            matched_key = key
            break
    
    if matched_key is not None:
        lr_min, lr_max, _ = OPTIMIZER_SEARCH_SPACES[matched_key]['lr']
    else:
        # Default to adaptive method range
        logger.warning(f"Optimizer '{optimizer_name}' not found in search spaces. Using default LR range.")
        lr_min, lr_max = 0.0001, 0.01
        
    return np.logspace(np.log10(lr_min), np.log10(lr_max), n_points).tolist()


def run_fair_lr_sweep(
    optimizers: List[str],
    train_fn: Callable,
    n_lr_points: int = 7,
    seeds: List[int] = None,
    save_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Run learning rate sweep with optimizer-appropriate ranges.
    
    This is Strategy B from HYPERPARAMETER_FAIRNESS_PROTOCOL.md:
    Fair comparison when full hyperparameter tuning is computationally prohibitive.
    
    Args:
        optimizers: List of optimizer names to compare
        train_fn: Function(optimizer_name, lr, seed) -> metrics_dict
        n_lr_points: Number of LR values to test per optimizer
        seeds: Random seeds for statistical validity (default: [42, 123, 456])
        save_dir: Directory to save results
        
    Returns:
        DataFrame with columns: [optimizer, lr, seed, <metrics>]
        
    Example:
        def my_train_fn(opt_name, lr, seed):
            model = create_model(seed)
            optimizer = create_optimizer(opt_name, lr)
            val_loss = train(model, optimizer)
            return {'val_loss': val_loss, 'val_acc': acc}
        
        results = run_fair_lr_sweep(
            optimizers=['SGD', 'Adam', 'AdamW'],
            train_fn=my_train_fn,
            n_lr_points=5,
            seeds=[42, 123, 456]
        )
    """
    if seeds is None:
        seeds = [42, 123, 456]
    
    logger.info(f"Starting fair LR sweep: {len(optimizers)} optimizers × {n_lr_points} LRs × {len(seeds)} seeds")
    logger.info(f"Total runs: {len(optimizers) * n_lr_points * len(seeds)}")
    
    results = []
    
    for opt_name in optimizers:
        lr_values = generate_lr_sweep(opt_name, n_lr_points)
        logger.info(f"\n{opt_name}: Testing LRs {[f'{lr:.6f}' for lr in lr_values]}")
        
        for lr in lr_values:
            for seed in seeds:
                logger.info(f"  Running {opt_name} lr={lr:.6f} seed={seed}")
                
                try:
                    metrics = train_fn(opt_name, lr, seed)
                    results.append({
                        'optimizer': opt_name,
                        'lr': lr,
                        'seed': seed,
                        **metrics
                    })
                except Exception as e:
                    logger.error(f"  Failed: {e}")
                    results.append({
                        'optimizer': opt_name,
                        'lr': lr,
                        'seed': seed,
                        'failed': True,
                        'error': str(e)
                    })
    
    df = pd.DataFrame(results)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir / 'fair_lr_sweep_results.csv', index=False)
        logger.info(f"Results saved to {save_dir / 'fair_lr_sweep_results.csv'}")
    
    return df


def select_best_lr_per_optimizer(
    results_df: pd.DataFrame,
    metric: str = 'val_loss',
    minimize: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    Select best learning rate for each optimizer based on validation performance.
    
    Returns mean ± std across seeds for the best LR.
    
    Args:
        results_df: DataFrame from run_fair_lr_sweep()
        metric: Metric to optimize (e.g., 'val_loss', 'val_accuracy')
        minimize: Whether to minimize or maximize metric
        
    Returns:
        Dict[optimizer_name] = (best_lr, best_metric_mean, best_metric_std)
    """
    best_configs = {}
    
    for opt_name in results_df['optimizer'].unique():
        opt_data = results_df[results_df['optimizer'] == opt_name]
        
        # Aggregate across seeds for each LR
        lr_aggregated = opt_data.groupby('lr')[metric].agg(['mean', 'std']).reset_index()
        
        # Select best LR
        if minimize:
            best_idx = lr_aggregated['mean'].idxmin()
        else:
            best_idx = lr_aggregated['mean'].idxmax()
        
        best_row = lr_aggregated.iloc[best_idx]
        best_configs[opt_name] = {
            'lr': best_row['lr'],
            'metric_mean': best_row['mean'],
            'metric_std': best_row['std']
        }
        
        logger.info(f"{opt_name}: Best LR = {best_row['lr']:.6f} "
                   f"({metric} = {best_row['mean']:.4f} ± {best_row['std']:.4f})")
    
    return best_configs


def compute_statistical_significance(
    results_df: pd.DataFrame,
    metric: str,
    baseline_optimizer: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute statistical significance of optimizer differences.
    
    Uses paired t-test (if normal) or Wilcoxon signed-rank test with
    Holm-Bonferroni correction for multiple comparisons.
    
    CRITICAL: Ensures paired samples are aligned by 'seed' for valid paired tests.
    
    Args:
        results_df: DataFrame with per-seed results for best configs
        metric: Metric to compare
        baseline_optimizer: Reference optimizer (e.g., 'SGD')
        alpha: Significance level (default 0.05)
        
    Returns:
        DataFrame with p-values, effect sizes, and significance flags
    """
    from scipy import stats
    
    comparisons = []
    
    # Extract baseline data with seed alignment
    baseline_df = results_df[results_df['optimizer'] == baseline_optimizer][['seed', metric]].copy()
    baseline_df = baseline_df.rename(columns={metric: 'baseline_metric'})
    
    for opt_name in results_df['optimizer'].unique():
        if opt_name == baseline_optimizer:
            continue
        
        # Extract optimizer data
        opt_df = results_df[results_df['optimizer'] == opt_name][['seed', metric]].copy()
        opt_df = opt_df.rename(columns={metric: 'opt_metric'})
        
        # CRITICAL FIX: Merge on seed to ensure paired samples are aligned
        merged = pd.merge(baseline_df, opt_df, on='seed', how='inner')
        
        if len(merged) < 3:
            logger.warning(
                f"Insufficient paired samples for {opt_name} vs {baseline_optimizer} "
                f"(n={len(merged)}). Skipping statistical test."
            )
            continue
        
        baseline_values = merged['baseline_metric'].values
        opt_values = merged['opt_metric'].values
        
        # Validate equal length (should be guaranteed by merge)
        assert len(baseline_values) == len(opt_values), "Sample length mismatch after merge"
        
        # Check normality with Shapiro-Wilk (handle small samples)
        try:
            _, p_norm_baseline = stats.shapiro(baseline_values)
            _, p_norm_opt = stats.shapiro(opt_values)
            is_normal = (p_norm_baseline > 0.05) and (p_norm_opt > 0.05)
        except Exception as e:
            logger.warning(f"Shapiro-Wilk failed for {opt_name}: {e}. Defaulting to non-parametric test.")
            is_normal = False
        
        # Paired test
        try:
            if is_normal:
                stat, p_value = stats.ttest_rel(baseline_values, opt_values)
                test_name = 'paired_t_test'
            else:
                stat, p_value = stats.wilcoxon(baseline_values, opt_values)
                test_name = 'wilcoxon'
        except Exception as e:
            logger.error(f"Statistical test failed for {opt_name}: {e}. Skipping.")
            continue
        
        # Effect size (Cohen's d for paired samples)
        diff = baseline_values - opt_values
        cohens_d = np.mean(diff) / max(np.std(diff, ddof=1), 1e-10)  # Avoid division by zero
        
        comparisons.append({
            'optimizer': opt_name,
            'baseline': baseline_optimizer,
            'test': test_name,
            'n_seeds': len(merged),
            'p_value': p_value,
            'cohens_d': cohens_d,
            'baseline_mean': np.mean(baseline_values),
            'optimizer_mean': np.mean(opt_values),
            'improvement': np.mean(baseline_values) - np.mean(opt_values)
        })
    
    if not comparisons:
        logger.warning("No valid comparisons computed. Returning empty DataFrame.")
        return pd.DataFrame()
    
    comp_df = pd.DataFrame(comparisons)
    
    # Apply Holm-Bonferroni correction with step-down procedure
    comp_df = comp_df.sort_values('p_value').reset_index(drop=True)
    n = len(comp_df)
    
    # CRITICAL FIX: Implement proper step-down procedure
    # Holm-Bonferroni is a sequential rejection procedure:
    # - Sort p-values from smallest to largest
    # - Test each p-value against alpha/(n-rank+1)
    # - STOP at the first non-rejection (all remaining are non-significant)
    comp_df['rank'] = range(1, n + 1)
    comp_df['holm_bonferroni_threshold'] = alpha / (n - comp_df['rank'] + 1)
    
    # Step-down logic: mark as significant only up to first non-rejection
    significant_flags = [False] * n
    for i in range(n):
        if comp_df.loc[i, 'p_value'] < comp_df.loc[i, 'holm_bonferroni_threshold']:
            significant_flags[i] = True
        else:
            # First non-rejection: stop here (all remaining are non-significant)
            break
    
    comp_df['significant_corrected'] = significant_flags
    
    return comp_df


def save_fairness_report(
    results_df: pd.DataFrame,
    best_configs: Dict,
    significance_df: Optional[pd.DataFrame],
    save_path: Path
):
    """
    Save comprehensive fairness report documenting experimental protocol.
    
    This ensures transparency and reproducibility per protocol requirements.
    """
    report = {
        'protocol': 'HYPERPARAMETER_FAIRNESS_PROTOCOL',
        'strategy': 'Fair LR Sweep (Strategy B)',
        'optimizers': list(best_configs.keys()),
        'best_hyperparameters': best_configs,
        'seeds': results_df['seed'].unique().tolist(),
        'n_seeds': len(results_df['seed'].unique()),
        'search_spaces': {
            opt: OPTIMIZER_SEARCH_SPACES.get(opt.upper(), 'custom')
            for opt in best_configs.keys()
        },
        'statistical_tests': significance_df.to_dict('records') if significance_df is not None else None,
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Fairness report saved to {save_path}")


if __name__ == '__main__':
    # Example usage
    print("Fair Optimizer Ablation Framework")
    print("=" * 60)
    print("\nOptimizer-Specific LR Ranges:")
    for opt, params in OPTIMIZER_SEARCH_SPACES.items():
        lr_range = params['lr']
        print(f"  {opt:15s}: LR ∈ [{lr_range[0]:.4f}, {lr_range[1]:.2f}] ({lr_range[2]})")
    
    print("\n\nExample LR Sweeps:")
    for opt in ['SGD', 'Adam', 'AdamW']:
        lrs = generate_lr_sweep(opt, n_points=5)
        print(f"  {opt:10s}: {[f'{lr:.6f}' for lr in lrs]}")
