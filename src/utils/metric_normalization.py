"""
Metric normalization utilities for backward compatibility.

Handles inconsistent metric naming across experiment outputs
(test_acc vs test_accuracy, val_acc vs val_accuracy, etc.)

AUDIT FIX: Addresses metric naming inconsistencies found during audit.
"""

from typing import Union, List, Optional
import pandas as pd
import logging


# Metric aliases mapping (preferred_name -> list of aliases)
METRIC_ALIASES = {
    'test_accuracy': ['test_acc', 'final_test_acc', 'final_test_accuracy'],
    'test_loss': ['test_loss', 'final_test_loss'],
    'val_accuracy': ['val_acc', 'validation_accuracy', 'validation_acc'],
    'val_loss': ['val_loss', 'validation_loss'],
    'train_accuracy': ['train_acc', 'training_accuracy', 'training_acc'],
    'train_loss': ['train_loss', 'training_loss'],
}


def normalize_metric_name(metric: str) -> str:
    """
    Normalize metric name to preferred standard.

    Args:
        metric: Metric name (potentially non-standard)

    Returns:
        Standardized metric name

    Example:
        >>> normalize_metric_name('test_acc')
        'test_accuracy'
        >>> normalize_metric_name('val_acc')
        'val_accuracy'
    """
    # Already standard
    if metric in METRIC_ALIASES:
        return metric

    # Search aliases
    for standard, aliases in METRIC_ALIASES.items():
        if metric in aliases:
            return standard

    # Unknown metric, return as-is
    return metric


def get_metric_column(df: pd.DataFrame, metric: str, strict: bool = False) -> Optional[str]:
    """
    Find actual column name for a metric, accounting for aliases.

    Args:
        df: DataFrame to search
        metric: Desired metric (preferred or alias)
        strict: If True, raise KeyError if not found. If False, return None.

    Returns:
        Actual column name in DataFrame, or None if not found

    Raises:
        KeyError: If strict=True and metric not found

    Example:
        >>> df = pd.DataFrame({'test_acc': [0.9], 'train_loss': [0.1]})
        >>> get_metric_column(df, 'test_accuracy')
        'test_acc'
    """
    # Try exact match first
    if metric in df.columns:
        return metric

    # Normalize and try again
    standard = normalize_metric_name(metric)
    if standard in df.columns:
        return standard

    # Try all aliases
    if standard in METRIC_ALIASES:
        for alias in METRIC_ALIASES[standard]:
            # Use scalar comparison to avoid pandas ambiguity
            if isinstance(df.columns, pd.Index) and alias in df.columns:
                logging.debug("Found metric '%s' as alias '%s'", metric, alias)
                return alias

    # Not found
    if strict:
        available = ', '.join(df.columns)
        raise KeyError(f"Metric '{metric}' not found in DataFrame. Available columns: {available}")

    return None


def extract_metric(df: pd.DataFrame, metric: str, default=None) -> Union[float, pd.Series, None]:
    """
    Extract metric value from DataFrame, handling aliases automatically.

    For single-row DataFrames, returns scalar value.
    For multi-row DataFrames, returns Series.

    Args:
        df: DataFrame containing metrics
        metric: Metric name (standard or alias)
        default: Value to return if metric not found

    Returns:
        Metric value(s) or default

    Example:
        >>> df = pd.DataFrame({'test_acc': [0.92]})
        >>> extract_metric(df, 'test_accuracy')
        0.92
    """
    col = get_metric_column(df, metric, strict=False)

    if col is None:
        return default

    values = df[col]

    # If we accidentally got a DataFrame slice, collapse it when possible
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 1:
            # Convert single-column DataFrame into a Series using column label (avoids tuple indexing)
            col_label = values.columns[0]
            # Column access returns a Series for a single-column DataFrame; silence indexing overload false-positive
            values = values[col_label]  # type: ignore[index]
        else:
            # Ambiguous multi-column selection: return default to be safe
            return default

    # Return scalar for single-value Series (coerce numpy scalars to Python floats)
    if isinstance(values, pd.Series) and len(values) == 1:
        v = values.iloc[0]
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    # Guard: if we still have a DataFrame here for some reason, treat as ambiguous and return default
    if isinstance(values, pd.DataFrame):
        return default

    return values


def normalize_dataframe_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Rename all metric columns to standard names.

    Args:
        df: DataFrame with potentially non-standard column names
        inplace: If True, modify DataFrame in-place

    Returns:
        DataFrame with standardized column names

    Example:
        >>> df = pd.DataFrame({'test_acc': [0.9], 'val_acc': [0.85]})
        >>> normalized = normalize_dataframe_columns(df)
        >>> list(normalized.columns)
        ['test_accuracy', 'val_accuracy']
    """
    if not inplace:
        df = df.copy()

    # Build rename mapping
    rename_map = {}
    for col in df.columns:
        standard = normalize_metric_name(col)
        if standard != col:
            rename_map[col] = standard

    if rename_map:
        logging.debug("Normalizing columns: %s", rename_map)
        df.rename(columns=rename_map, inplace=True)

    return df


def aggregate_metric_across_seeds(
    dfs: List[pd.DataFrame],
    metric: str,
    phase: str = 'eval',
    aggregation: str = 'last'
) -> List[float]:
    """
    Extract and aggregate metric across multiple seed runs.

    Args:
        dfs: List of DataFrames (one per seed)
        metric: Metric to extract
        phase: Phase to filter on (e.g., 'eval', 'train')
        aggregation: How to aggregate per-run ('last', 'max', 'mean')

    Returns:
        List of metric values (one per seed)

    Example:
        >>> df1 = pd.DataFrame({'phase': ['eval'], 'test_acc': [0.91]})
        >>> df2 = pd.DataFrame({'phase': ['eval'], 'test_acc': [0.93]})
        >>> aggregate_metric_across_seeds([df1, df2], 'test_accuracy')
        [0.91, 0.93]
    """
    values = []

    for df in dfs:
        # Filter by phase if column exists
        if 'phase' in df.columns:
            df_phase = df[df['phase'] == phase]
            if len(df_phase) == 0:
                logging.warning("No rows with phase='%s', using full DataFrame", phase)
                df_phase = df
        else:
            df_phase = df

        # Ensure we have a DataFrame for downstream helpers
        if not isinstance(df_phase, pd.DataFrame):
            df_phase = pd.DataFrame(df_phase)

        # Get metric column
        col = get_metric_column(df_phase, metric, strict=False)

        if col is None:
            logging.warning("Metric '%s' not found in DataFrame", metric)
            continue

        # Aggregate
        metric_values = df_phase[col]
        # If numpy array slipped through, wrap as Series to allow dropna
        if not hasattr(metric_values, 'dropna'):
            metric_values = pd.Series(metric_values)
        metric_values = metric_values.dropna()

        if len(metric_values) == 0:
            continue

        if aggregation == 'last':
            values.append(float(metric_values.iloc[-1]))
        elif aggregation == 'max':
            values.append(float(metric_values.max()))
        elif aggregation == 'mean':
            values.append(float(metric_values.mean()))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return values


if __name__ == '__main__':
    # Self-test (np imported only for test)
    import numpy as _np_test

    print("Testing metric normalization utilities...")

    # Test normalize_metric_name
    assert normalize_metric_name('test_acc') == 'test_accuracy'
    assert normalize_metric_name('val_acc') == 'val_accuracy'
    assert normalize_metric_name('test_accuracy') == 'test_accuracy'
    print("✅ normalize_metric_name")

    # Test get_metric_column
    test_df = pd.DataFrame({'test_acc': [0.9], 'train_loss': [0.1]})
    assert get_metric_column(test_df, 'test_accuracy') == 'test_acc'
    assert get_metric_column(test_df, 'unknown_metric') is None
    print("✅ get_metric_column")

    # Test extract_metric
    result = extract_metric(test_df, 'test_accuracy')
    # Handle both scalar and Series results
    if isinstance(result, pd.Series):
        result_val = float(result.iloc[0])
    else:
        result_val = float(result) if result is not None else None
    assert result_val is not None and result_val == 0.9
    assert extract_metric(test_df, 'unknown') is None
    print("✅ extract_metric")

    # Test normalize_dataframe_columns
    df_norm = normalize_dataframe_columns(test_df)
    assert 'test_accuracy' in df_norm.columns
    assert 'test_acc' not in df_norm.columns
    print("✅ normalize_dataframe_columns")

    print("\n✅ All tests passed!")
