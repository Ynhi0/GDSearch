"""
CSV utilities: safe_read_csv provides robust, import-safe CSV handling suitable for
batch experiments and notebook workflows.

Features:
- Accepts str or pathlib.Path input
- Returns pd.DataFrame on success, None on empty CSVs
- Raises ValueError for missing or malformed files with clear messages
- Provides `cleanup_empty_csvs` to quarantine bad CSVs
"""
from pathlib import Path
from typing import Optional
import logging
import pandas as pd


class CSVReadError(Exception):
    """Raised when a CSV cannot be read due to I/O or parsing issues."""


def safe_read_csv(path: str | Path, *, header_required: bool = True, **kwargs) -> Optional[pd.DataFrame]:
    """Safely read a CSV file.

    Args:
        path: File path to read.
        header_required: If True, treat files without a detectable header as invalid and return None.
        **kwargs: Passed to pandas.read_csv when reading data (applied for full read only).

    Returns:
        pd.DataFrame on success, None if file is empty or header is missing (when required).

    Raises:
        CSVReadError on I/O or parser errors (useful for callers that want to fail fast).
    """
    p = Path(path)

    if not p.exists():
        logging.debug("CSV path does not exist: %s", p)
        raise CSVReadError(f"CSV file does not exist: {p}")

    # Fast sanity checks
    try:
        size = p.stat().st_size
    except OSError as e:
        logging.exception("Could not stat CSV '%s': %s", p, e)
        raise CSVReadError(f"Could not stat CSV '{p}': {e}") from e

    if size == 0:
        logging.warning("CSV file '%s' is empty (size=0).", p)
        return None

    # Try reading a single row to validate parseability and presence of header
    # EXPLICIT context manager ensures file is closed
    try:
        with open(p, 'r', encoding='utf-8', newline='') as f:
            sample = pd.read_csv(f, nrows=1)
    except pd.errors.EmptyDataError:
        logging.warning("CSV file '%s' raised EmptyDataError on sample read.", p)
        return None
    except pd.errors.ParserError as e:
        logging.exception("CSV file '%s' parser error on sample read: %s", p, e)
        raise CSVReadError(f"Parser error while reading CSV '{p}': {e}") from e
    except (OSError, UnicodeDecodeError, ValueError) as e:
        logging.exception("Unexpected error while sampling CSV '%s': %s", p, e)
        raise CSVReadError(f"Unexpected error while sampling CSV '{p}': {e}") from e
    except Exception:
        # Surface unexpected exceptions during development
        raise

    # If header is required but sample has no columns, treat as invalid
    if header_required and (sample.columns is None or len(sample.columns) == 0):
        logging.warning("CSV file '%s' appears headerless and 'header_required' is True.", p)
        return None

    # Full read with provided kwargs
    # EXPLICIT context manager ensures file is closed
    try:
        with open(p, 'r', encoding='utf-8', newline='') as f:
            df = pd.read_csv(f, **kwargs)
        if df is None or df.shape[0] == 0:
            logging.warning("CSV '%s' yielded zero rows after full read.", p)
            return None
        return df
    except pd.errors.EmptyDataError:
        logging.warning("CSV file '%s' became empty during full read.", p)
        return None
    except pd.errors.ParserError as e:
        logging.exception("Parser error reading CSV '%s': %s", p, e)
        raise CSVReadError(f"Parser error while reading CSV '{p}': {e}") from e
    except Exception as e:
        logging.exception("Failed to read CSV '%s': %s", p, e)
        raise CSVReadError(f"Failed to read CSV '{p}': {e}") from e


def cleanup_empty_csvs(results_dir: str | Path, pattern: str = "*.csv") -> list:
    """Scan results directory for empty or unreadable CSV files and move them to a `corrupt/` subdirectory.

    Args:
        results_dir: Path to scan (directory containing CSV files).
        pattern: Glob pattern for CSV files to consider.

    Returns:
        List of paths (as strings) that were moved to the corrupt folder.
    """
    import shutil

    moved = []
    base = Path(results_dir)
    corrupt_dir = base / "corrupt"
    corrupt_dir.mkdir(parents=True, exist_ok=True)

    for p in base.rglob(pattern):
        # Skip files that are already in the corrupt quarantine to avoid re-processing
        if corrupt_dir in p.parents:
            continue
        # BUG FIX: Only process files, not directories (even if they match pattern)
        if not p.is_file():
            continue
        # Additional safety: skip if path doesn't exist anymore (race condition)
        if not p.exists():
            continue
        try:
            # If file is empty by size, consider it corrupt
            if p.stat().st_size == 0:
                target = corrupt_dir / p.name
                shutil.move(str(p), str(target))
                logging.warning("Moved empty CSV '%s' to corrupt folder", p)
                moved.append(str(target))
                continue

            # Try validating one row using safe_read_csv behavior
            try:
                if safe_read_csv(p, header_required=False, nrows=1) is None:
                    target = corrupt_dir / p.name
                    shutil.move(str(p), str(target))
                    logging.warning("Moved unreadable/empty CSV '%s' to corrupt folder", p)
                    moved.append(str(target))
            except CSVReadError:
                target = corrupt_dir / p.name
                shutil.move(str(p), str(target))
                logging.warning("Moved CSV with read error '%s' to corrupt folder", p)
                moved.append(str(target))
        except (OSError, shutil.Error, PermissionError) as e:
            logging.exception("Failed while inspecting '%s': %s", p, e)
    return moved
