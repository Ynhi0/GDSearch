"""
Metric Aggregation Utilities

Provides robust metric aggregation that handles NaN values correctly.

Bug Fix: Prevents NaN propagation when aggregating metrics across runs.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple runs, filtering NaN values.
    
    Bug Fix: NaN in one metric no longer contaminates all aggregations.
    Each metric is aggregated independently with NaN filtering.
    
    Args:
        metrics_list: List of metric dictionaries from different runs
        
    Returns:
        Dictionary with aggregated metrics (mean values)
        
    Example:
        >>> run1 = {'accuracy': 0.95, 'loss': 0.1}
        >>> run2 = {'accuracy': np.nan, 'loss': 0.12}
        >>> run3 = {'accuracy': 0.94, 'loss': 0.11}
        >>> aggregate_metrics([run1, run2, run3])
        {'accuracy': 0.945, 'loss': 0.11}  # accuracy excludes NaN from run2
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all unique keys across all metrics
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = []
        for metrics in metrics_list:
            if key in metrics:
                values.append(metrics[key])
        
        # Filter NaN values
        valid_values = []
        for v in values:
            if isinstance(v, (int, float)):
                if not np.isnan(v):
                    valid_values.append(float(v))
            else:
                # Non-numeric values - keep as-is (for strings, etc.)
                valid_values.append(v)
        
        if not valid_values:
            # All values were NaN - keep NaN
            aggregated[key] = np.nan
        elif all(isinstance(v, (int, float)) for v in valid_values):
            # Numeric values - compute mean
            aggregated[key] = float(np.mean(valid_values))
            
            # Log warning if NaN values were filtered
            if len(valid_values) < len(values):
                logging.warning(
                    f"Metric '{key}': {len(values) - len(valid_values)}/{len(values)} "
                    f"NaN values filtered during aggregation"
                )
        else:
            # Mixed types or non-numeric - take first value
            aggregated[key] = valid_values[0]
            if len(set(str(v) for v in valid_values)) > 1:
                logging.warning(
                    f"Metric '{key}': Multiple distinct non-numeric values. Using first: {valid_values[0]}"
                )
    
    return aggregated


def aggregate_with_std(metrics_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics with mean and standard deviation.
    
    Args:
        metrics_list: List of metric dictionaries from different runs
        
    Returns:
        Dictionary mapping metric names to {'mean': float, 'std': float, 'count': int}
        
    Example:
        >>> run1 = {'accuracy': 0.95, 'loss': 0.1}
        >>> run2 = {'accuracy': 0.94, 'loss': 0.12}
        >>> aggregate_with_std([run1, run2])
        {
            'accuracy': {'mean': 0.945, 'std': 0.005, 'count': 2},
            'loss': {'mean': 0.11, 'std': 0.01, 'count': 2}
        }
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = []
        for metrics in metrics_list:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    values.append(float(val))
        
        if not values:
            aggregated[key] = {
                'mean': np.nan,
                'std': np.nan,
                'count': 0
            }
        elif len(values) == 1:
            aggregated[key] = {
                'mean': values[0],
                'std': 0.0,
                'count': 1
            }
        else:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'count': len(values)
            }
    
    return aggregated


def safe_metric_value(value: Any, default: float = np.nan) -> float:
    """
    Safely convert metric value to float, handling NaN and non-numeric values.
    
    Args:
        value: Metric value to convert
        default: Default value to return if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except (ValueError, TypeError):
        return default


def filter_valid_metrics(metrics: Dict[str, Any], 
                        required_keys: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Filter out metric dictionaries with invalid/missing values.
    
    Args:
        metrics: Dictionary of metrics
        required_keys: Optional list of required keys. If any is missing/invalid, return None
        
    Returns:
        Filtered metrics dictionary or None if critical values are missing
        
    Example:
        >>> metrics = {'accuracy': 0.95, 'loss': np.nan, 'epoch': 10}
        >>> filter_valid_metrics(metrics, required_keys=['accuracy'])
        {'accuracy': 0.95, 'loss': nan, 'epoch': 10}  # passes because 'accuracy' is valid
        >>> filter_valid_metrics(metrics, required_keys=['loss'])
        None  # fails because required 'loss' is NaN
    """
    if not metrics:
        return None
    
    if required_keys:
        for key in required_keys:
            if key not in metrics:
                logging.debug(f"Missing required key: {key}")
                return None
            
            val = metrics[key]
            if isinstance(val, (int, float)):
                if np.isnan(val) or np.isinf(val):
                    logging.debug(f"Invalid value for required key '{key}': {val}")
                    return None
    
    return metrics
