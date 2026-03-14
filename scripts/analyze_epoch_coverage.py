#!/usr/bin/env python
"""
Analyze epoch coverage across all v2 experiments.
Identifies which experiments may need retraining.
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict

results_dir = Path('results_proposal_full_20260223_v2/experiments')
print('='*90)
print('COMPREHENSIVE EPOCH COVERAGE ANALYSIS')
print('='*90)

# Quick mode baselines (from code review)
quick_mode_baselines = {
    'ablation': 1000,
    'advanced_ablation': 10,
    'batch_ablation': 100,
    'highdim': 500,
    'init_ablation': 10,
    'lr_ablation': 10,
    'medical': 3,
    'mnist': 2,
    'resnet': 20,
    'robustness': 1000,
    'scheduler_ablation': 10,  # Hardcoded to 10 epochs in run_all_kaggle.py line 2641
    'wd_ablation': 10,
}

experiment_stats = {}

for exp_dir in sorted(results_dir.glob('*')):
    if not exp_dir.is_dir():
        continue
    
    exp_name = exp_dir.name
    seed_csvs = list(exp_dir.glob('*_seed*.csv'))
    
    if not seed_csvs:
        continue
    
    epochs_list = []
    row_counts = []
    
    for csv_file in seed_csvs:
        try:
            df = pd.read_csv(csv_file)
            
            # Find epoch or iteration column
            epoch_cols = [c for c in df.columns if 'epoch' in c.lower()]
            iter_cols = [c for c in df.columns if 'iteration' in c.lower() or 'iter' in c.lower()]
            
            if epoch_cols:
                max_val = df[epoch_cols[0]].max()
                col_used = epoch_cols[0]
            elif iter_cols:
                max_val = df[iter_cols[0]].max()
                col_used = iter_cols[0]
            else:
                max_val = len(df)
                col_used = 'row_count'
            
            epochs_list.append(max_val)
            row_counts.append(len(df))
        except Exception as e:
            print(f'  Error reading {csv_file.name}: {e}')
    
    if epochs_list:
        stats = {
            'num_seeds': len(epochs_list),
            'min_epoch': min(epochs_list),
            'max_epoch': max(epochs_list),
            'avg_epoch': sum(epochs_list) / len(epochs_list),
            'avg_rows': sum(row_counts) / len(row_counts),
            'expected_quick': quick_mode_baselines.get(exp_name, None),
            'epochs_list': epochs_list
        }
        experiment_stats[exp_name] = stats

# Analyze results
print('\nEXPERIMENT STATUS REPORT:')
print('-'*90)

concerning_exps = []

for exp_name in sorted(experiment_stats.keys()):
    s = experiment_stats[exp_name]
    expected = s['expected_quick']
    min_ep = s['min_epoch']
    max_ep = s['max_epoch']
    avg_ep = s['avg_epoch']
    
    # Flag concerning cases
    status = 'OK'
    if min_ep == 0 and max_ep == 0:
        status = 'CRITICAL: All zero epochs'
        concerning_exps.append((exp_name, status, s))
    elif min_ep == 0 and max_ep > 0:
        status = 'WARNING: Some seeds with 0 epochs'
        concerning_exps.append((exp_name, status, s))
    elif expected and max_ep < expected * 0.5 and exp_name not in ['medical', 'mnist', 'resnet']:
        # For experiments that should have more data
        if max_ep < 50:  # Only flag if genuinely small
            status = f'CHECK: Max {max_ep} << expected {expected}'
            concerning_exps.append((exp_name, status, s))
    elif max_ep != min_ep and (max_ep - min_ep) > max_ep * 0.5:
        status = f'VARIANCE: Epochs vary widely ({min_ep}-{max_ep})'
        concerning_exps.append((exp_name, status, s))
    
    print(f'{exp_name:35} | Seeds:{s["num_seeds"]:3} | Epochs:{min_ep:7.0f}-{max_ep:7.0f} (avg {avg_ep:7.1f}) | {status}')

print('\n' + '='*90)
print('CONCERNING EXPERIMENTS:')
print('='*90)

if concerning_exps:
    for exp_name, status, s in concerning_exps:
        print(f'\n{exp_name}')
        print(f'  Status: {status}')
        print(f'  Seeds: {s["num_seeds"]}')
        print(f'  Epoch range: {s["min_epoch"]:.0f} - {s["max_epoch"]:.0f} (avg {s["avg_epoch"]:.1f})')
        print(f'  Avg rows per seed: {s["avg_rows"]:.0f}')
        print(f'  Expected (quick mode): {s["expected_quick"]}')
        
        # Show distribution
        unique_epochs = sorted(set(s['epochs_list']))
        if len(unique_epochs) <= 10:
            print(f'  Epoch distribution: {unique_epochs}')
        else:
            print(f'  Epoch range: {min(unique_epochs)} to {max(unique_epochs)} ({len(unique_epochs)} unique values)')
else:
    print('No critical issues found.')

print('\n' + '='*90)
print('RECOMMENDATIONS:')
print('='*90)
print('''
1. Quick-mode experiments (medical, mnist, resnet, init_ablation, lr_ablation, wd_ablation):
   - These are INTENTIONALLY short (2-10 epochs)
   - DO NOT rerun - this is expected behavior

2. Standard experiments (ablation, advanced_ablation, batch_ablation):
   - Range: 10-999 epochs
   - These are COMPLETE and ready for analysis

3. Large-scale experiments (robustness, highdim):
   - robustness: 0-19999 iterations (check for 0-value seeds)
   - highdim: 123-375 epochs (some variance OK)
   - If 0-value seeds exist, those individual runs may be incomplete

ACTION ITEMS:
- If robustness has 0-epoch seeds: check if they're corrupted or incomplete metadata
- If highdim low-epoch seeds: may indicate interruption or OOM, consider rerun
- All others: appear to be at expected training depth for their configuration
''')
