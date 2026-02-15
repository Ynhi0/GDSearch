"""
Filesystem safety utilities for experiment runs.

Provides defensive checks for disk space, permissions, and cleanup operations
to prevent experiment failures after hours of computation.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Union
import time
import uuid


def check_write_permission(path: Union[str, Path]) -> bool:
    """
    Check if path is writable before starting experiment.
    
    Creates parent directories if needed and tests write access
    with a temporary file.
    
    Args:
        path: File path to check (can be file or directory)
    
    Returns:
        True if writable, False otherwise
    
    Example:
        >>> results_path = Path("results/experiment.csv")
        >>> if not check_write_permission(results_path):
        ...     raise PermissionError("Cannot write results!")
    """
    path = Path(path)
    
    # Determine directory to check
    if path.is_dir() or str(path).endswith('/'):
        check_dir = path
    else:
        check_dir = path.parent
    
    try:
        # Create directory if it doesn't exist
        check_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to create and remove a test file
        test_file = check_dir / f".write_test_{uuid.uuid4()}.tmp"
        
        try:
            test_file.touch()
            test_file.unlink()
            logging.debug(f"Write permission check passed: {check_dir}")
            return True
        except (OSError, PermissionError) as e:
            logging.error(f"Cannot write to {check_dir}: {e}")
            return False
        
    except (OSError, PermissionError) as e:
        logging.error(f"Cannot create directory {check_dir}: {e}")
        return False


def check_disk_space(
    path: Union[str, Path],
    required_mb: float = 500,
    check_type: str = "checkpoint"
) -> bool:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Path to check (directory or file)
        required_mb: Required free space in MB
        check_type: Type of operation for error messages ("checkpoint", "results", etc.)
    
    Returns:
        True if sufficient space available, False otherwise
    
    Example:
        >>> if not check_disk_space("./checkpoints", required_mb=1000):
        ...     logging.error("Not enough disk space for checkpoints!")
    """
    path = Path(path)
    
    # Get parent directory if path is a file
    if not path.is_dir():
        check_dir = path.parent
    else:
        check_dir = path
    
    try:
        # Get disk usage statistics
        stat = shutil.disk_usage(check_dir)
        free_mb = stat.free / (1024 * 1024)
        total_mb = stat.total / (1024 * 1024)
        used_pct = 100 * (1 - stat.free / stat.total)
        
        if free_mb < required_mb:
            logging.error(
                f"Insufficient disk space for {check_type}:\n"
                f"  Location: {check_dir}\n"
                f"  Free: {free_mb:.0f} MB ({100-used_pct:.1f}% of {total_mb:.0f} MB)\n"
                f"  Required: {required_mb:.0f} MB\n"
                f"  Shortfall: {required_mb - free_mb:.0f} MB\n"
                f"REMEDIATION:\n"
                f"  1. Free up disk space (delete old files, clear cache)\n"
                f"  2. Move experiment to different location with more space\n"
                f"  3. Reduce checkpoint frequency\n"
                f"  4. Use compressed checkpoints"
            )
            return False
        
        # Warn if disk is very full (>90%)
        if used_pct > 90:
            logging.warning(
                f"Disk is {used_pct:.1f}% full at {check_dir}. "
                f"Only {free_mb:.0f} MB free. "
                f"Monitor disk usage during experiment."
            )
        
        logging.debug(
            f"Disk space check passed: {free_mb:.0f} MB free (>= {required_mb:.0f} MB required)"
        )
        return True
    
    except OSError as e:
        logging.warning(f"Could not check disk space for {check_dir}: {e}")
        # Assume OK if check fails (fail open) but log warning
        return True


def cleanup_stale_temp_files(
    base_dir: Union[str, Path],
    max_age_hours: int = 24,
    pattern: str = "**/*.tmp",
    dry_run: bool = False
) -> int:
    """
    Remove stale temporary files older than max_age_hours.
    
    This should be called periodically (e.g., at checkpoint manager init)
    to prevent accumulation of failed save attempts.
    
    Args:
        base_dir: Directory to search for temp files
        max_age_hours: Maximum age in hours before file is considered stale
        pattern: Glob pattern for temp files (default: **/*.tmp)
        dry_run: If True, only log what would be deleted without deleting
    
    Returns:
        Number of files cleaned up
    
    Example:
        >>> # Clean up temp files older than 1 day
        >>> n_cleaned = cleanup_stale_temp_files("./checkpoints", max_age_hours=24)
        >>> print(f"Removed {n_cleaned} stale temp files")
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        logging.debug(f"Directory does not exist: {base_dir}")
        return 0
    
    cutoff_time = time.time() - (max_age_hours * 3600)
    cleaned_count = 0
    
    try:
        for temp_file in base_dir.glob(pattern):
            if not temp_file.is_file():
                continue
            
            try:
                # Check file age
                mtime = temp_file.stat().st_mtime
                age_hours = (time.time() - mtime) / 3600
                
                if mtime < cutoff_time:
                    if dry_run:
                        logging.info(
                            f"Would remove stale temp file (age: {age_hours:.1f}h): {temp_file}"
                        )
                    else:
                        logging.info(
                            f"Removing stale temp file (age: {age_hours:.1f}h): {temp_file}"
                        )
                        temp_file.unlink()
                    
                    cleaned_count += 1
            
            except OSError as e:
                logging.debug(f"Could not process temp file {temp_file}: {e}")
                continue
        
        if cleaned_count > 0:
            action = "Would remove" if dry_run else "Removed"
            logging.info(f"{action} {cleaned_count} stale temp file(s) from {base_dir}")
        else:
            logging.debug(f"No stale temp files found in {base_dir}")
    
    except Exception as e:
        logging.warning(f"Error during temp file cleanup: {e}")
    
    return cleaned_count


def ensure_directory_exists(path: Union[str, Path], check_writable: bool = True) -> Path:
    """
    Ensure directory exists and is writable.
    
    Args:
        path: Directory path
        check_writable: If True, verify write permission
    
    Returns:
        Path object (guaranteed to exist and be writable if check_writable=True)
    
    Raises:
        PermissionError: If directory cannot be created or is not writable
    
    Example:
        >>> results_dir = ensure_directory_exists("results/experiment_1")
    """
    path = Path(path)
    
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise PermissionError(
            f"Cannot create directory {path}: {e}\n"
            f"REMEDIATION:\n"
            f"  1. Check parent directory permissions\n"
            f"  2. Verify filesystem is not read-only\n"
            f"  3. Use different output location"
        ) from e
    
    if check_writable and not check_write_permission(path):
        raise PermissionError(
            f"Directory {path} exists but is not writable.\n"
            f"REMEDIATION:\n"
            f"  1. Check directory permissions: ls -ld {path}\n"
            f"  2. Change permissions: chmod u+w {path}\n"
            f"  3. Use different output location"
        )
    
    return path


def safe_remove_file(path: Union[str, Path], missing_ok: bool = True) -> bool:
    """
    Safely remove file with error handling.
    
    Args:
        path: File to remove
        missing_ok: If True, don't raise error if file doesn't exist
    
    Returns:
        True if file was removed, False otherwise
    
    Example:
        >>> safe_remove_file("old_checkpoint.pth")
    """
    path = Path(path)
    
    if not path.exists():
        if missing_ok:
            return False
        else:
            raise FileNotFoundError(f"File not found: {path}")
    
    try:
        path.unlink()
        logging.debug(f"Removed file: {path}")
        return True
    except OSError as e:
        logging.warning(f"Could not remove file {path}: {e}")
        return False


def get_directory_size(path: Union[str, Path]) -> float:
    """
    Calculate total size of directory in MB.
    
    Args:
        path: Directory path
    
    Returns:
        Size in MB
    
    Example:
        >>> size_mb = get_directory_size("./checkpoints")
        >>> print(f"Checkpoints use {size_mb:.1f} MB")
    """
    path = Path(path)
    
    if not path.exists():
        return 0.0
    
    total_bytes = 0
    
    try:
        for item in path.rglob('*'):
            if item.is_file():
                try:
                    total_bytes += item.stat().st_size
                except OSError:
                    # Skip files we can't stat
                    continue
    except Exception as e:
        logging.warning(f"Error calculating directory size for {path}: {e}")
    
    return total_bytes / (1024 * 1024)


def monitor_disk_usage(
    paths: list[Union[str, Path]],
    warn_threshold_pct: float = 90.0,
    error_threshold_pct: float = 95.0
) -> dict[str, dict[str, float]]:
    """
    Monitor disk usage for multiple paths and log warnings.
    
    Args:
        paths: List of paths to monitor
        warn_threshold_pct: Log warning if disk usage exceeds this percent
        error_threshold_pct: Log error if disk usage exceeds this percent
    
    Returns:
        Dict mapping path -> {free_mb, total_mb, used_pct}
    
    Example:
        >>> stats = monitor_disk_usage([
        ...     "./checkpoints",
        ...     "./results",
        ...     "./artifacts"
        ... ])
    """
    usage_stats = {}
    
    for path in paths:
        path = Path(path)
        
        # Get parent directory if path is a file
        if not path.is_dir():
            check_dir = path.parent
        else:
            check_dir = path
        
        try:
            stat = shutil.disk_usage(check_dir)
            free_mb = stat.free / (1024 * 1024)
            total_mb = stat.total / (1024 * 1024)
            used_pct = 100 * (1 - stat.free / stat.total)
            
            usage_stats[str(path)] = {
                'free_mb': free_mb,
                'total_mb': total_mb,
                'used_pct': used_pct
            }
            
            # Log warnings based on thresholds
            if used_pct >= error_threshold_pct:
                logging.error(
                    f"CRITICAL: Disk {used_pct:.1f}% full at {check_dir}. "
                    f"Only {free_mb:.0f} MB free. Experiment may fail!"
                )
            elif used_pct >= warn_threshold_pct:
                logging.warning(
                    f"WARNING: Disk {used_pct:.1f}% full at {check_dir}. "
                    f"{free_mb:.0f} MB free."
                )
        
        except OSError as e:
            logging.debug(f"Could not check disk usage for {check_dir}: {e}")
    
    return usage_stats
