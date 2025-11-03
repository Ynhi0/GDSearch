"""
Baseline Comparison: Compare custom optimizers with PyTorch implementations.

This script compares our custom optimizer implementations with:
1. PyTorch's built-in optimizers (torch.optim)
2. Published benchmarks from papers (if available)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List
import matplotlib.pyplot as plt

from run_nn_experiment import run_nn_experiment, build_model_and_data
from statistical_analysis import compare_optimizers_ttest, print_ttest_results
from data_utils import get_mnist_loaders, get_cifar10_loaders


def run_pytorch_baseline(config: Dict) -> pd.DataFrame:
    """
    Run experiment with PyTorch's built-in optimizer for comparison.
    
    Args:
        config: Experiment configuration
        
    Returns:
        DataFrame with training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Build model and data
    model, train_loader, test_loader = build_model_and_data(
        dataset=config['dataset'],
        model_name=config['model'],
        batch_size=config.get('batch_size', 128),
        device=device
    )
    
    # Build PyTorch optimizer
    optimizer_name = config['optimizer']
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_name == 'Adam_PyTorch':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            weight_decay=weight_decay
        )
    elif optimizer_name == 'AdamW_PyTorch':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            weight_decay=weight_decay
        )
    elif optimizer_name == 'SGD_PyTorch':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.0),
            weight_decay=weight_decay
        )
    elif optimizer_name == 'RMSprop_PyTorch':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=config.get('alpha', 0.99),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown PyTorch optimizer: {optimizer_name}")
    
    criterion = nn.CrossEntropyLoss()
    epochs = config.get('epochs', 10)
    
    # Training loop
    history = []
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Compute gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = np.sqrt(grad_norm)
            
            optimizer.step()
            
            global_step += 1
            
            history.append({
                'phase': 'train',
                'epoch': epoch,
                'batch': batch_idx,
                'global_step': global_step,
                'train_loss': loss.item(),
                'grad_norm': grad_norm,
                'update_norm': 0.0  # Not tracked for PyTorch baseline
            })
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= len(test_loader)
        test_accuracy = correct / total
        
        history.append({
            'phase': 'eval',
            'epoch': epoch,
            'batch': 0,
            'global_step': global_step,
            'train_loss': 0.0,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'grad_norm': 0.0,
            'update_norm': 0.0
        })
    
    return pd.DataFrame(history)


def run_baseline_comparison(
    base_config: Dict,
    seeds: List[int],
    results_dir: str = 'results/baselines'
) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """
    Run comparison between custom and PyTorch baseline optimizers.
    
    Returns:
        Nested dict: {optimizer_type: {'custom': [dfs], 'pytorch': [dfs]}}
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("BASELINE COMPARISON: Custom vs PyTorch Optimizers")
    print("="*70)
    print(f"Dataset: {base_config.get('dataset')}")
    print(f"Model: {base_config.get('model')}")
    print(f"Seeds: {seeds}")
    print("="*70)
    
    # Optimizers to compare
    optimizer_pairs = [
        ('Adam', 'Adam_PyTorch'),
        ('AdamW', 'AdamW_PyTorch'),
        ('SGD_Momentum', 'SGD_PyTorch'),
        ('RMSProp', 'RMSprop_PyTorch')
    ]
    
    results = {}
    
    for custom_opt, pytorch_opt in optimizer_pairs:
        print(f"\n{'='*70}")
        print(f"Comparing: {custom_opt} vs {pytorch_opt}")
        print(f"{'='*70}")
        
        results[custom_opt] = {'custom': [], 'pytorch': []}
        
        for seed in seeds:
            # Custom optimizer
            print(f"\n  [{custom_opt}] Seed {seed}... ", end='', flush=True)
            
            custom_config = base_config.copy()
            custom_config['optimizer'] = custom_opt
            custom_config['seed'] = seed
            
            # Set optimizer-specific params
            if 'Adam' in custom_opt:
                custom_config['beta1'] = 0.9
                custom_config['beta2'] = 0.999
            elif 'Momentum' in custom_opt:
                custom_config['momentum'] = 0.9
            elif 'RMSProp' in custom_opt:
                custom_config['alpha'] = 0.99
            
            df_custom = run_nn_experiment(custom_config)
            results[custom_opt]['custom'].append(df_custom)
            
            # Save
            filename = f"{custom_opt}_custom_seed{seed}.csv"
            df_custom.to_csv(os.path.join(results_dir, filename), index=False)
            
            eval_df = df_custom[df_custom['phase'] == 'eval']
            if not eval_df.empty:
                final_acc = eval_df['test_accuracy'].iloc[-1]
                print(f"Acc: {final_acc:.4f}")
            else:
                print("Done")
            
            # PyTorch optimizer
            print(f"  [{pytorch_opt}] Seed {seed}... ", end='', flush=True)
            
            pytorch_config = base_config.copy()
            pytorch_config['optimizer'] = pytorch_opt
            pytorch_config['seed'] = seed
            
            # Copy params
            if 'Adam' in pytorch_opt:
                pytorch_config['beta1'] = 0.9
                pytorch_config['beta2'] = 0.999
            elif 'SGD' in pytorch_opt:
                pytorch_config['momentum'] = 0.9
            elif 'RMSprop' in pytorch_opt:
                pytorch_config['alpha'] = 0.99
            
            df_pytorch = run_pytorch_baseline(pytorch_config)
            results[custom_opt]['pytorch'].append(df_pytorch)
            
            # Save
            filename = f"{custom_opt}_pytorch_seed{seed}.csv"
            df_pytorch.to_csv(os.path.join(results_dir, filename), index=False)
            
            eval_df = df_pytorch[df_pytorch['phase'] == 'eval']
            if not eval_df.empty:
                final_acc = eval_df['test_accuracy'].iloc[-1]
                print(f"Acc: {final_acc:.4f}")
            else:
                print("Done")
    
    print("\n" + "="*70)
    print("✅ Baseline comparison completed!")
    print("="*70)
    
    return results


def analyze_baseline_comparison(results: Dict) -> pd.DataFrame:
    """Analyze baseline comparison results."""
    summary_data = []
    
    for optimizer_name, impl_dict in results.items():
        for impl_type, dfs in impl_dict.items():
            # Extract final accuracies
            final_accs = []
            for df in dfs:
                eval_df = df[df['phase'] == 'eval']
                if not eval_df.empty:
                    final_accs.append(eval_df['test_accuracy'].iloc[-1])
            
            if final_accs:
                summary_data.append({
                    'Optimizer': optimizer_name,
                    'Implementation': impl_type.capitalize(),
                    'Mean Accuracy': np.mean(final_accs),
                    'Std Accuracy': np.std(final_accs),
                    'N Seeds': len(final_accs)
                })
    
    return pd.DataFrame(summary_data)


def print_baseline_summary(summary_df: pd.DataFrame):
    """Print formatted baseline comparison summary."""
    print("\n" + "="*70)
    print("BASELINE COMPARISON RESULTS")
    print("="*70)
    
    optimizers = summary_df['Optimizer'].unique()
    
    for opt in optimizers:
        print(f"\n{opt}:")
        print("─"*70)
        
        opt_df = summary_df[summary_df['Optimizer'] == opt]
        
        custom_row = opt_df[opt_df['Implementation'] == 'Custom'].iloc[0]
        pytorch_row = opt_df[opt_df['Implementation'] == 'Pytorch'].iloc[0]
        
        print(f"  Custom:  {custom_row['Mean Accuracy']:.4f} ± {custom_row['Std Accuracy']:.4f}")
        print(f"  PyTorch: {pytorch_row['Mean Accuracy']:.4f} ± {pytorch_row['Std Accuracy']:.4f}")
        
        diff = custom_row['Mean Accuracy'] - pytorch_row['Mean Accuracy']
        diff_pct = (diff / pytorch_row['Mean Accuracy']) * 100
        
        if abs(diff) < 0.001:
            status = "✅ EQUIVALENT"
        elif diff > 0:
            status = f"✅ CUSTOM BETTER (+{diff:.4f}, {diff_pct:+.2f}%)"
        else:
            status = f"⚠️ PYTORCH BETTER ({diff:.4f}, {diff_pct:+.2f}%)"
        
        print(f"  → {status}")
    
    print("\n" + "="*70)


def plot_baseline_comparison(summary_df: pd.DataFrame, save_path: str = None):
    """Plot baseline comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    optimizers = summary_df['Optimizer'].unique()
    x = np.arange(len(optimizers))
    width = 0.35
    
    custom_means = []
    custom_stds = []
    pytorch_means = []
    pytorch_stds = []
    
    for opt in optimizers:
        opt_df = summary_df[summary_df['Optimizer'] == opt]
        
        custom_row = opt_df[opt_df['Implementation'] == 'Custom'].iloc[0]
        pytorch_row = opt_df[opt_df['Implementation'] == 'Pytorch'].iloc[0]
        
        custom_means.append(custom_row['Mean Accuracy'])
        custom_stds.append(custom_row['Std Accuracy'])
        pytorch_means.append(pytorch_row['Mean Accuracy'])
        pytorch_stds.append(pytorch_row['Std Accuracy'])
    
    # Bars
    bars1 = ax.bar(x - width/2, custom_means, width, yerr=custom_stds,
                   label='Custom', capsize=8, alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, pytorch_means, width, yerr=pytorch_stds,
                   label='PyTorch', capsize=8, alpha=0.8, color='#e74c3c')
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers, fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Custom vs PyTorch Optimizer Implementations\n(Mean ± Std across seeds)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Baseline comparison plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def perform_statistical_tests(results: Dict):
    """Perform statistical tests between custom and PyTorch implementations."""
    print("\n" + "="*70)
    print("STATISTICAL TESTS (Custom vs PyTorch)")
    print("="*70)
    
    for optimizer_name, impl_dict in results.items():
        custom_accs = []
        pytorch_accs = []
        
        for df in impl_dict['custom']:
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                custom_accs.append(eval_df['test_accuracy'].iloc[-1])
        
        for df in impl_dict['pytorch']:
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                pytorch_accs.append(eval_df['test_accuracy'].iloc[-1])
        
        if custom_accs and pytorch_accs:
            result = compare_optimizers_ttest(
                np.array(custom_accs),
                np.array(pytorch_accs),
                name_A=f"{optimizer_name} (Custom)",
                name_B=f"{optimizer_name} (PyTorch)",
                metric='test_accuracy'
            )
            
            print_ttest_results(result)


def main():
    """Run full baseline comparison."""
    
    # Base configuration
    base_config = {
        'dataset': 'MNIST',
        'model': 'SimpleMLP',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 10,
        'batch_size': 128
    }
    
    seeds = [1, 2, 3]
    
    # Run comparison
    results = run_baseline_comparison(base_config, seeds)
    
    # Analyze
    summary_df = analyze_baseline_comparison(results)
    
    # Print
    print_baseline_summary(summary_df)
    
    # Save
    os.makedirs('results/baselines', exist_ok=True)
    summary_df.to_csv('results/baselines/baseline_comparison.csv', index=False)
    
    # Plot
    os.makedirs('plots', exist_ok=True)
    plot_baseline_comparison(summary_df, save_path='plots/baseline_comparison.png')
    
    # Statistical tests
    perform_statistical_tests(results)
    
    print("\n" + "="*70)
    print("✅ BASELINE COMPARISON COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
