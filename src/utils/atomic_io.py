"""
Atomic file operations for safe CSV writes.

Prevents corruption from crashes, OOM errors, and interrupted writes.
Uses temp file + atomic rename pattern.
"""

from pathlib import Path
from typing import Union, Any
import pandas as pd
import logging


def safe_write_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    Write CSV file atomically using temp file + rename.
    
    This prevents corruption from crashes/OOM by:
    1. Writing to a temporary file first
    2. Only replacing the target file if write succeeds
    3. Atomic rename operation (POSIX) or near-atomic (Windows)
    
    Args:
        df: DataFrame to save
        path: Target file path
        **kwargs: Additional arguments for pd.DataFrame.to_csv
    
    Raises:
        OSError: If write or rename fails
    
    Example:
        safe_write_csv(results_df, 'results/experiment.csv', index=False)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use .tmp extension to avoid confusion with partial writes
    temp_path = path.with_suffix('.csv.tmp')
    
    try:
        # Write to temp file
        df.to_csv(temp_path, **kwargs)
        
        # Atomic rename (POSIX) or near-atomic (Windows)
        # On POSIX: rename() is atomic if source and dest are on same filesystem
        # On Windows: replace() is near-atomic (very small race window)
        temp_path.replace(path)
        
        logging.debug(f"Atomically wrote CSV: {path}")
        
    except Exception as e:
        # Clean up temp file on failure
        try:
            temp_path.unlink(missing_ok=True)
        except Exception as cleanup_error:
            logging.debug(f"Failed to clean up temp file {temp_path}: {cleanup_error}")
        
        # Re-raise original error
        raise OSError(f"Failed to write CSV to {path}: {e}") from e


def safe_write_json(data: Any, path: Union[str, Path], **kwargs) -> None:
    """
    Write JSON file atomically using temp file + rename.
    
    Args:
        data: Data to serialize as JSON
        path: Target file path
        **kwargs: Additional arguments for json.dump (e.g., indent=2)
    """
    import json
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = path.with_suffix('.json.tmp')
    
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **kwargs)
        
        temp_path.replace(path)
        logging.debug(f"Atomically wrote JSON: {path}")
        
    except Exception as e:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise OSError(f"Failed to write JSON to {path}: {e}") from e


def safe_write_text(text: str, path: Union[str, Path], encoding: str = 'utf-8') -> None:
    """
    Write text file atomically using temp file + rename.
    
    Args:
        text: Text content to write
        path: Target file path
        encoding: Text encoding (default: utf-8)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = path.with_suffix('.txt.tmp')
    
    try:
        with open(temp_path, 'w', encoding=encoding) as f:
            f.write(text)
        
        temp_path.replace(path)
        logging.debug(f"Atomically wrote text: {path}")
        
    except Exception as e:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise OSError(f"Failed to write text to {path}: {e}") from e


# Backward compatibility alias
def safe_to_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Alias for safe_write_csv for backward compatibility."""
    # Remove 'index' from kwargs if present and set default to False
    if 'index' not in kwargs:
        kwargs['index'] = False
    safe_write_csv(df, path, **kwargs)
