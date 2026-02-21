import math
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from ..utils.metric_normalization import to_percent_series, to_percent
from ..utils.filename import parse_experiment_filename
from contextlib import contextmanager
import logging


@contextmanager
def plot_protect(log_on_fail: bool = True, strict: bool = False, logger: logging.Logger | None = None):
    """Context manager to protect plotting code from failing the whole run.

    Usage:
        with plot_protect(log_on_fail=True):
            plt.savefig(...)

    By default catches plotting-related errors and logs a single WARNING. If
    ``strict=True`` the exception is re-raised (useful for CI/debugging).

    Note: This intentionally uses a broad catch to isolate visual failures
    from the rest of the run; keep narrow where possible outside of plotting.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        yield
    except (OSError, RuntimeError, ValueError) as e:
        # These are expected plotting/IO/runtime errors we can reasonably
        # catch and continue from during normal runs.
        if strict:
            logger.warning("Plot failed (strict mode): re-raising: %s", e)
            raise
        if log_on_fail:
            logger.warning("Plotting failed: %s", e)
            logger.debug("Plotting failure details:", exc_info=True)
    except Exception as e:  # broad catch intentional: isolate any plotting error
        if strict:
            logger.warning("Plot failed (strict mode): re-raising unexpected exception: %s", e)
            raise
        if log_on_fail:
            logger.warning("Plotting failed (unexpected): %s", e)
            logger.debug("Unexpected plotting failure:", exc_info=True)


def filter_time_series_files(csv_paths: List[Path]) -> List[Path]:
    """Return subset of csv_paths that appear to be time-series (contain 'epoch' column)"""
    ts = []
    for p in csv_paths:
        try:
            # read only header to check columns
            df = pd.read_csv(p, nrows=1)
            if 'epoch' in df.columns or 'iteration' in df.columns:
                ts.append(p)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError):
            # Skip files that cannot be read
            continue
        except Exception:
            # Broad catch intentional: unexpected errors while scanning CSVs should not break visualization discovery
            continue
    return ts


def safe_add_text(ax, x, y, text, **kwargs):
    """Safely add text to a matplotlib axis only if coordinates are finite."""
    try:
        x_finite = np.isfinite(x)
        y_finite = np.isfinite(y)
    except (TypeError, ValueError):
        x_finite = False
        y_finite = False
    except Exception:
        # Broad catch intentional: guard against unexpected numeric types
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
