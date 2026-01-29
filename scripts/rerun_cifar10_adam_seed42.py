"""Run CIFAR10 Adam seed=42 single-run, convert to legacy CSV naming, and update summary & visuals."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.run_cifar10 import run_single
import pandas as pd
from src.utils.metric_normalization import to_percent
from src.utils.filename import parse_opt_seed_from_stem

results_dir = Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10')
results_dir.mkdir(parents=True, exist_ok=True)

# Run a single Adam experiment
out = run_single('Adam', seed=42, lr=1e-3, epochs=50, batch_size=128, results_dir=results_dir)
print('Produced:', out)

# Load produced CSV and convert to legacy naming and percent scale
df = pd.read_csv(out)
# If test_accuracy exists as fraction 0-1, convert to percent
if 'test_accuracy' in df.columns:
    vals = pd.to_numeric(df['test_accuracy'], errors='coerce')
    if not vals.isna().all() and vals.max() <= 1.01:
        df['test_accuracy'] = vals * 100.0
# Build legacy filename: CIFAR10_ResNet18_Adam_seed42.csv
legacy = results_dir / 'CIFAR10_ResNet18_Adam_seed42.csv'
# Map columns to legacy format: test_acc column expected
if 'test_accuracy' in df.columns and 'test_acc' not in df.columns:
    df = df.rename(columns={'test_accuracy': 'test_acc'})

# Ensure final row has test_acc
if 'test_acc' in df.columns:
    df['test_acc'] = pd.to_numeric(df['test_acc'], errors='coerce')

# Save legacy CSV (only epoch-level columns present in other files)
df.to_csv(legacy, index=False)
print('Saved legacy CSV:', legacy)

# Update summary CSV (CIFAR10_summary.csv) by recomputing from per-run CSVs
# Recompute summary from all CIFAR10_ResNet18_*.csv files
files = list(results_dir.glob('CIFAR10_ResNet18_*.csv'))
rows = []
for f in files:
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    # Use robust parser for optimizer and seed
    stem = f.stem
    opt, seed = parse_opt_seed_from_stem(stem)
    # last non-null test_acc
    acc_col = None
    for c in ['final_test_acc', 'test_acc', 'test_accuracy', 'val_acc']:
        if c in d.columns:
            acc_col = c
            break
    final = None
    if acc_col:
        s = d[acc_col].dropna()
        if len(s) > 0:
            val = s.iloc[-1]
            final = to_percent(val)
    # Only add row if we have a valid optimizer and final metric
    if opt and final is not None and not pd.isna(final):
        rows.append({'optimizer': opt, 'seed': int(seed) if seed is not None else 0, 'final_test_acc': final})

if rows:
    sdf = pd.DataFrame(rows)
    sdf_grp = sdf.groupby('optimizer')['final_test_acc'].agg(['mean','std']).reset_index()
    # Write CIFAR10_summary.csv
    summary_path = results_dir / 'CIFAR10_summary.csv'
    # Save detailed per-seed too
    sdf.to_csv(results_dir / 'CIFAR10_summary_per_seed.csv', index=False)
    sdf_grp.to_csv(summary_path, index=False)
    print('Updated summary:', summary_path)

# Regenerate visuals for CIFAR10
from runners.run_all_kaggle import create_experiment_visualizations
csvs = list(results_dir.glob('*.csv'))
create_experiment_visualizations('CIFAR10', r'C:/Users/MPhuc/Downloads/results/results_full', csvs)
print('Regenerated visuals')
