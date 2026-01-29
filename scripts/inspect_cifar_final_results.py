#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

results_dir = Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10')
files = list(results_dir.glob('*.csv'))
print('Found', len(files), 'csv files')

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        # try to infer optimizer/seed
        stem = f.stem
        parts = stem.split('_')
        if 'optimizer' not in df.columns:
            for i, part in enumerate(parts):
                if 'seed' in part and i > 0:
                    df['optimizer'] = parts[i-1]
                    break
        if 'seed' not in df.columns:
            for part in parts:
                if 'seed' in part:
                    try:
                        df['seed'] = int(part.replace('seed',''))
                    except Exception:
                        df['seed'] = 0
                    break
        dfs.append(df)
    except Exception as e:
        print('skip', f.name, e)

if not dfs:
    print('No CSVs loaded')
    raise SystemExit(1)

combined = pd.concat(dfs, ignore_index=True)
print('Combined rows:', combined.shape[0])
print('Columns:', combined.columns.tolist())

# choose accuracy column
acc_cols = ['final_test_acc','test_acc','test_accuracy','val_acc','test_accuracy']
acc_col=None
for c in acc_cols:
    if c in combined.columns:
        acc_col=c
        break
print('Using acc column:', acc_col)

last = combined.sort_values('epoch').groupby(['optimizer','seed']).last()
print('\nPer-optimizer sample of last-per-seed values:')
for opt in sorted(set(last.index.get_level_values(0))):
    sub = last.loc[opt]
    if acc_col in sub.columns:
        vals = sub[acc_col].dropna()
        print(opt, 'count', len(vals), 'max', vals.max(), 'min', vals.min(), 'dtype', vals.dtype)

# final aggregated means
if acc_col in last.columns:
    per_opt = last.reset_index().groupby('optimizer')[acc_col].agg(['mean','std'])
    print('\nPer-opt aggregated (mean,std):')
    print(per_opt)
    try:
        final_max = float(per_opt['mean'].abs().max())
        print('final_max', final_max)
    except Exception as e:
        print('error computing final_max', e)

print('\nDone')
