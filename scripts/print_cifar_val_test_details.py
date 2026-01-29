"""Print detailed per-file presence and sample values for test/val loss and accuracy."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.utils.metric_normalization import normalize_dataframe_columns, to_percent_series

p = Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10')
files = sorted([f for f in p.glob('*.csv') if 'summary' not in f.name.lower() and 'results' not in f.name.lower()])
print('File, test_loss_count, last_test_loss, val_loss_count, last_val_loss, test_accuracy_count, last_test_accuracy, val_accuracy_count, last_val_accuracy')
for f in files:
    df = pd.read_csv(f)
    df = normalize_dataframe_columns(df, inplace=False)
    # Ensure val/test columns are normalized
    for col in ['test_loss','test_accuracy','val_loss','val_accuracy','test_acc','val_acc']:
        if col in df.columns:
            pass
    # Fill derived if necessary
    if 'test_loss' not in df.columns and 'val_loss' in df.columns:
        df['test_loss'] = df['val_loss']
    if 'test_accuracy' not in df.columns:
        if 'val_accuracy' in df.columns:
            df['test_accuracy'] = df['val_accuracy']
        elif 'val_acc' in df.columns:
            df['test_accuracy'] = df['val_acc']
    # normalize test_accuracy to percent
    if 'test_accuracy' in df.columns:
        df['test_accuracy'] = to_percent_series(pd.to_numeric(df['test_accuracy'], errors='coerce'))
    # counts and last values
    def last_or_none(s):
        s = pd.to_numeric(s, errors='coerce').dropna()
        if len(s)==0:
            return 0, None
        return int(len(s)), float(s.iloc[-1])

    tl_cnt, tl_last = last_or_none(df['test_loss']) if 'test_loss' in df.columns else (0, None)
    vl_cnt, vl_last = last_or_none(df['val_loss']) if 'val_loss' in df.columns else (0, None)
    ta_cnt, ta_last = last_or_none(df['test_accuracy']) if 'test_accuracy' in df.columns else (0, None)
    va_cnt, va_last = last_or_none(df['val_accuracy']) if 'val_accuracy' in df.columns else (0, None)

    print(f.name, tl_cnt, tl_last, vl_cnt, vl_last, ta_cnt, ta_last, va_cnt, va_last)