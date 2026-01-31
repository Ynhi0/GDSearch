"""
File I/O safety utilities for GDSearch experiments.

Ensures robust file operations with automatic directory creation
and error handling to prevent common FileNotFoundError issues.
"""

from pathlib import Path
from typing import Union
import pandas as pd
import logging


def ensure_parent_dir(filepath: Union[str, Path]) -> Path:
    """
    Ensure parent directory exists for a given filepath.

    Args:
        filepath: Path to file (can be string or Path object)

    Returns:
        Path object with guaranteed parent directory existence

    Example:
        >>> path = ensure_parent_dir("results/subdir/output.csv")
        >>> # results/subdir/ now exists
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def safe_to_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> Path:
    """
    Save DataFrame to CSV with automatic directory creation.

    AUDIT FIX: Prevents FileNotFoundError by ensuring parent directories exist.
    Use this instead of df.to_csv() in experiment code.

    Args:
        df: DataFrame to save
        filepath: Destination path
        **kwargs: Additional arguments passed to df.to_csv()

    Returns:
        Path object where file was saved

    Example:
        >>> results = pd.DataFrame({'loss': [0.5, 0.3, 0.1]})
        >>> safe_to_csv(results, "artifacts/experiment/run1.csv", index=False)
    """
    path = ensure_parent_dir(filepath)
    try:
        df.to_csv(path, **kwargs)
        return path
    except (OSError, PermissionError, ValueError) as e:
        logging.exception("Failed to save CSV to %s: %s", path, e)
        raise


def safe_write_text(content: str, filepath: Union[str, Path], encoding: str = 'utf-8') -> Path:
    """
    Write text to file with automatic directory creation.

    Args:
        content: Text content to write
        filepath: Destination path
        encoding: Text encoding (default: utf-8)

    Returns:
        Path object where file was written
    """
    path = ensure_parent_dir(filepath)
    try:
        path.write_text(content, encoding=encoding)
        return path
    except (OSError, PermissionError, ValueError) as e:
        logging.exception("Failed to write text to %s: %s", path, e)
        raise
