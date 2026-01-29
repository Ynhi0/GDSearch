"""Print a brief summary for each CIFAR10 CSV: rows, final accuracy column present, final value (raw and normalized).
"""
import pathlib
import sys
# Ensure project root is on path for imports
sys.path.insert(0, r'C:/Users/MPhuc/Desktop/GDSearch')
import pandas as pd
from src.utils.metric_normalization import to_percent

p = pathlib.Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10')
files = sorted([f for f in p.glob('*.csv') if 'summary' not in f.name.lower() and 'results' not in f.name.lower()])
print('Found', len(files), 'csv files')

values = []
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f.name, 'ERROR reading:', e)
        continue
    if df.empty:
        print(f.name, 'EMPTY')
        continue
    # Candidate columns
    acc_cols = ['final_test_acc', 'test_acc', 'test_accuracy', 'val_acc']
    col = None
    for c in acc_cols:
        if c in df.columns:
            col = c
            break
    rows = len(df)
    last_val = None
    if col is not None:
        # try to find last non-null entry for that column
        s = df[col].dropna()
        if len(s) > 0:
            last_val = s.iloc[-1]
    # fallback: check for a final_ column specific
    if last_val is None and 'final_test_acc' in df.columns:
        last_val = df['final_test_acc'].dropna().iloc[-1] if df['final_test_acc'].dropna().size > 0 else None

    normalized = None
    if last_val is not None:
        normalized = to_percent(last_val)
        values.append((f.name, normalized))

    print(f"{f.name}: rows={rows}, acc_col={col}, raw_last={last_val}, normalized={normalized}")

# Aggregate per optimizer (parse optimizer from filename: CIFAR10_ResNet18_<OPT>_seed...
from collections import defaultdict
agg = defaultdict(list)
for name, val in values:
    # split by underscores
    parts = name.split('_')
    # optimizer is the 3rd component after CIFAR10, ResNet18
    if len(parts) >= 4:
        opt = parts[2]
    else:
        opt = 'UNKNOWN'
    agg[opt].append(val)

print('\nPer-optimizer aggregates:')
for opt, vals in sorted(agg.items()):
    import math
    cnt = len(vals)
    mean = sum(vals) / cnt if cnt else float('nan')
    var = sum((v - mean) ** 2 for v in vals) / cnt if cnt else float('nan')
    std = math.sqrt(var) if cnt else float('nan')
    print(f"{opt}: count={cnt}, mean={mean:.2f}, std={std:.2f}")
