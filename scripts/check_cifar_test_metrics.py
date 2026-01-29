"""Check CIFAR10 per-run CSVs for presence of test_loss/test_accuracy values after normalization."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metric_normalization import normalize_dataframe_columns, to_percent_series
import pandas as pd

p = Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10')
files = sorted([f for f in p.glob('*.csv') if 'summary' not in f.name.lower() and 'results' not in f.name.lower()])
print('Checking', len(files), 'files')
for f in files:
    df = pd.read_csv(f)
    df = normalize_dataframe_columns(df, inplace=False)
    # fallback and fill empty
    if 'test_loss' not in df.columns and 'val_loss' in df.columns:
        df['test_loss'] = df['val_loss']
    if 'test_loss' in df.columns and df['test_loss'].dropna().empty and 'val_loss' in df.columns:
        df['test_loss'] = df['val_loss']

    if 'test_accuracy' not in df.columns:
        if 'val_accuracy' in df.columns:
            df['test_accuracy'] = df['val_accuracy']
        elif 'val_acc' in df.columns:
            df['test_accuracy'] = df['val_acc']
    if 'test_accuracy' in df.columns and df['test_accuracy'].dropna().empty:
        if 'val_accuracy' in df.columns:
            df['test_accuracy'] = df['val_accuracy']
        elif 'val_acc' in df.columns:
            df['test_accuracy'] = df['val_acc']
    if 'test_accuracy' in df.columns:
        df['test_accuracy'] = to_percent_series(pd.to_numeric(df['test_accuracy'], errors='coerce'))

    print(f.name, 'test_loss nonnull', df['test_loss'].dropna().shape[0] if 'test_loss' in df.columns else 0,
          'test_accuracy nonnull', df['test_accuracy'].dropna().shape[0] if 'test_accuracy' in df.columns else 0)