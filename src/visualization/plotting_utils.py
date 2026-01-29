import math
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from ..utils.metric_normalization import to_percent_series, to_percent
from ..utils.filename import parse_experiment_filename


def filter_time_series_files(csv_paths: List[Path]) -> List[Path]:
    """Return subset of csv_paths that appear to be time-series (contain 'epoch' column)"""
    ts = []
    for p in csv_paths:
        try:
            # read only header to check columns
            df = pd.read_csv(p, nrows=1)
            if 'epoch' in df.columns:
                ts.append(p)
        except Exception:
            # Skip files that cannot be read
            continue
    return ts


def safe_add_text(ax, x, y, text, **kwargs):
    """Safely add text to a matplotlib axis only if coordinates are finite."""
    try:
        x_finite = np.isfinite(x)
        y_finite = np.isfinite(y)
    except Exception:
        x_finite = False
        y_finite = False

    if x_finite and y_finite:
        ax.text(x, y, text, **kwargs)


def normalize_final_results(series_or_df):
    """Normalize final results into a DataFrame with mean/std (percents, 0-100) and drop non-finite rows."""
    if isinstance(series_or_df, pd.Series):
        df = series_or_df.to_frame('mean')
        df['std'] = 0.0
    else:
        df = series_or_df.copy()

    # Prefer columns: mean, std
    if 'mean' not in df.columns and 'final' in df.columns:
        df = df.rename(columns={'final': 'mean'})

    # Apply percent normalization
    df['mean'] = to_percent_series(df['mean'])
    if 'std' in df.columns:
        df['std'] = df['std'].apply(lambda x: to_percent(x))
    else:
        df['std'] = 0.0

    # Cap to 0..100 and drop NaN/non-finite
    df['mean'] = df['mean'].clip(lower=0.0, upper=100.0)
    df['std'] = df['std'].clip(lower=0.0, upper=100.0)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df
