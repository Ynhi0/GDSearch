#!/usr/bin/env python
"""Inspect a 0-epoch robustness seed file."""
import pandas as pd
from pathlib import Path

robustness_dir = Path('results_proposal_full_20260223_v2/experiments/robustness')

# Check one of the 0-epoch files
zero_file = robustness_dir / 'Robustness_Rosenbrock_Adam_start7_seed42.csv'

print('Examining a 0-epoch seed file:')
print('='*80)
print(f'File: {zero_file.name}')
print(f'File exists: {zero_file.exists()}')
print(f'File size: {zero_file.stat().st_size} bytes')

df = pd.read_csv(zero_file)
print(f'Rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(f'Column dtypes:\n{df.dtypes}')
print(f'\nFirst 5 rows:')
print(df.head())

# Compare with a normal file
normal_file = robustness_dir / 'Robustness_Rosenbrock_Adam_start6_seed42.csv'
if normal_file.exists():
    print('\n' + '='*80)
    print(f'Comparing with normal file: {normal_file.name}')
    df_normal = pd.read_csv(normal_file)
    print(f'Rows: {len(df_normal)}')
    print(f'Columns: {df_normal.columns.tolist()}')
    if len(df_normal) > 0:
        print(f'Iteration range: {df_normal.iloc[:, 0].min()} to {df_normal.iloc[:, 0].max()}')
    print(f'Rows match: {len(df_normal) == len(df)}')
else:
    print(f'\nNo comparison file found. Searching for similar...')
    for f in list(robustness_dir.glob('*_seed42.csv'))[:3]:
        df_tmp = pd.read_csv(f)
        print(f'  {f.name}: {len(df_tmp)} rows')
