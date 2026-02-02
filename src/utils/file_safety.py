"""
File I/O safety utilities for GDSearch experiments.

Ensures robust file operations with automatic directory creation,
atomic writes, and error handling.

CRITICAL FIX: Now uses atomic writes to prevent corruption from
crashes, OOM errors, and interrupted writes.
"""

from pathlib import Path
from typing import Union
import pandas as pd
import logging

# Import atomic write functions
from src.utils.atomic_io import safe_write_csv as _atomic_write_csv
from src.utils.atomic_io import safe_write_text as _atomic_write_text
from src.utils.atomic_io import safe_write_json


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
    Save DataFrame to CSV with automatic directory creation and atomic writes.

    CRITICAL FIX: Now uses atomic temp-file + rename to prevent corruption.
    Previous version used direct df.to_csv() which could create partial files
    on crashes/OOM, breaking resume logic.

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
    path = Path(filepath)
    
    # Delegate to atomic write function
    try:
        _atomic_write_csv(df, path, **kwargs)
        return path
    except (OSError, PermissionError, ValueError) as e:
        logging.exception("Failed to atomically save CSV to %s: %s", path, e)
        raise


def safe_write_text(content: str, filepath: Union[str, Path], encoding: str = 'utf-8') -> Path:
    """
    Write text to file with automatic directory creation and atomic writes.

    CRITICAL FIX: Now uses atomic temp-file + rename to prevent corruption.

    Args:
        content: Text content to write
        filepath: Destination path
        encoding: Text encoding (default: utf-8)

    Returns:
        Path object where file was written
    """
    path = Path(filepath)
    
    try:
        _atomic_write_text(content, path, encoding=encoding)
        return path
    except (OSError, PermissionError, ValueError) as e:
        logging.exception("Failed to atomically write text to %s: %s", path, e)
        raise

