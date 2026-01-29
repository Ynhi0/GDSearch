"""
CSV utilities: include a safe_read_csv wrapper which returns None for empty CSV files
instead of raising, so notebooks and batch scripts can skip empty CSVs gracefully.
"""
from typing import Optional
import logging
import pandas as pd


def safe_read_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """Read CSV and return DataFrame or None for empty files.

    Returns:
        pd.DataFrame if file has data, None if file is empty.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        logging.warning("CSV file '%s' is empty. Skipping.", path)
        return None
    except Exception:
        logging.exception("Failed to read CSV '%s'", path)
        raise


def cleanup_empty_csvs(results_dir: str | 'Path', pattern: str = "*.csv") -> list:
    """Scan results directory for empty or unreadable CSV files and move them to a `corrupt/` subdirectory.

    Args:
        results_dir: Path to scan (directory containing CSV files).
        pattern: Glob pattern for CSV files to consider.

    Returns:
        List of paths (as strings) that were moved to the corrupt folder.
    """
    from pathlib import Path
    import shutil
    moved = []
    base = Path(results_dir)
    corrupt_dir = base / "corrupt"
    corrupt_dir.mkdir(parents=True, exist_ok=True)

    for p in base.rglob(pattern):
        # Only process files
        if not p.is_file():
            continue
        try:
            # If file is empty by size, consider it corrupt
            if p.stat().st_size == 0:
                target = corrupt_dir / p.name
                shutil.move(str(p), str(target))
                logging.warning("Moved empty CSV '%s' to corrupt folder", p)
                moved.append(str(target))
                continue

            # Try reading a single row to detect parser errors / truly empty data
            try:
                pd.read_csv(p, nrows=1)
            except pd.errors.EmptyDataError:
                target = corrupt_dir / p.name
                shutil.move(str(p), str(target))
                logging.warning("Moved unreadable/empty CSV '%s' to corrupt folder", p)
                moved.append(str(target))
            except Exception:
                # For any other read error, log and move the file as well
                target = corrupt_dir / p.name
                shutil.move(str(p), str(target))
                logging.warning("Moved CSV with read error '%s' to corrupt folder", p)
                moved.append(str(target))
        except Exception:
            logging.exception("Failed while inspecting '%s'", p)
    return moved
