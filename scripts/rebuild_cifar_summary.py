"""Rebuild CIFAR10 summary CSVs from per-run CSVs without running experiments.

This script:
 - Scans experiments/cifar10 for CSVs named CIFAR10_ResNet18_*.csv and NN_ResNet18_*.csv
 - Normalizes test accuracy columns (fractions -> percent)
 - Parses optimizer and seed using parse_opt_seed_from_stem
 - Writes CIFAR10_summary_per_seed.csv and CIFAR10_summary.csv
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.utils.filename import parse_opt_seed_from_stem
from src.utils.metric_normalization import to_percent

results_dir = Path(r"C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10")
files = list(results_dir.glob('CIFAR10_ResNet18_*.csv')) + list(results_dir.glob('NN_ResNet18_*.csv'))
rows = []
for f in files:
    try:
        d = pd.read_csv(f)
    except Exception as e:
        print('skip', f, 'err', e)
        continue
    stem = f.stem
    opt, seed = parse_opt_seed_from_stem(stem)
    # detect accuracy column
    acc_col = None
    for c in ['final_test_acc', 'test_acc', 'test_accuracy', 'val_acc']:
        if c in d.columns:
            acc_col = c
            break
    if acc_col is None:
        continue
    s = pd.to_numeric(d[acc_col], errors='coerce').dropna()
    if len(s) == 0:
        continue
    # if values look like fractions (max <= 1.01), convert to percent
    if s.max() <= 1.01:
        s = s * 100.0
    final = s.iloc[-1]
    if opt and final is not None and pd.notna(final):
        rows.append({'optimizer': opt, 'seed': int(seed) if seed is not None else 0, 'final_test_acc': float(final)})

if rows:
    sdf = pd.DataFrame(rows)
    sdf.to_csv(results_dir / 'CIFAR10_summary_per_seed.csv', index=False)
    sdf_grp = sdf.groupby('optimizer')['final_test_acc'].agg(['mean','std']).reset_index()
    sdf_grp.to_csv(results_dir / 'CIFAR10_summary.csv', index=False)
    print('Wrote summaries to', results_dir)
else:
    print('No rows to summarize')