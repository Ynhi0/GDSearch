"""
Robust checkpoint management with backup, validation, and disk space awareness.

This module provides the RobustCheckpointManager class for production-grade
checkpoint handling with atomic writes, rollback, and integrity validation.

Locking protocol (implemented in _create_backup):
- Locks are implemented via lock files created atomically with open(..., 'x').
- Each locker writes a unique token (``pid:uuid4``) into the lock file; only
  the creator that holds the matching token may remove the lock.
- If a lock file exists and is younger than ``stale_lock_seconds`` we wait up to
  ``backup_lock_timeout`` seconds for it to be released. If the lock is older
  than ``stale_lock_seconds`` it is considered stale and may be removed
  (best-effort) to prevent deadlocks from crashed processes.
- This design avoids accidental unlocking by other processes while allowing
  safe stale-lock recovery when necessary.
"""
import os
import time
import random
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Import compatibility helpers for cross-version torch I/O
try:
    from src.core.io_utils import torch_load_safe, torch_save_safe
except (ImportError, AttributeError) as e:
    # If centralized I/O helpers are not available, keep local fallbacks, but warn explicitly.
    logging.warning("Could not import src.core.io_utils; using local I/O helpers: %s", e)

    def torch_load_safe(path_or_file, map_location=None, weights_only=None):
        try:
            if weights_only is not None:
                return torch.load(path_or_file, map_location=map_location, weights_only=weights_only)
            else:
                return torch.load(path_or_file, map_location=map_location)
        except TypeError:
            return torch.load(path_or_file, map_location=map_location)

    def torch_save_safe(obj, path_or_file, use_new_zipfile_serialization=True):
        try:
            if use_new_zipfile_serialization:
                torch.save(obj, path_or_file, _use_new_zipfile_serialization=True)
            else:
                torch.save(obj, path_or_file)
        except TypeError:
            torch.save(obj, path_or_file)


class RobustCheckpointManager:
    """
    Robust checkpointing with backup, validation, and disk space awareness.

    Features:
    - Atomic writes with temp file + fsync + rename
    - Rolling backups (configurable)
    - Checkpoint validation
    - RNG state capture for reproducibility
    - Disk space monitoring
    """

    def __init__(self, base_dir: str, max_backups: int = 1, min_free_gb: float = 1.0, strict: bool = True, backup_lock_timeout: int = 30, stale_lock_seconds: int = 3600):
        """
        Initialize checkpoint manager.

        Args:
            base_dir: Base directory for checkpoints
            max_backups: Maximum number of backup files to keep
            min_free_gb: Minimum free disk space in GB
            strict: If True, raise on critical save/load/validation failures. If False, return False and log warnings (backward compatibility).
            backup_lock_timeout: Time in seconds to wait for an existing lock before timing out.
            stale_lock_seconds: Age in seconds after which an existing lock is considered stale and may be removed.
        """
        self.base_dir = Path(base_dir)
        self.max_backups = max_backups
        self.min_free_gb = min_free_gb
        self.strict = bool(strict)
        self.backup_lock_timeout = int(backup_lock_timeout)
        self.stale_lock_seconds = int(stale_lock_seconds)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize disk space guardian if available
        self._disk_guardian = None
        try:
            from src.core.training_enhancements import DiskSpaceGuardian
            self._disk_guardian = DiskSpaceGuardian(
                self.base_dir,
                min_free_gb=min_free_gb,
                max_checkpoints=max_backups * 3
            )
        except (ImportError, AttributeError) as e:
            logging.debug("DiskSpaceGuardian not available: %s", e)

    def save_checkpoint(
        self,
        checkpoint_data: Dict,
        filename: Optional[str],
        experiment_name: Optional[str]
    ) -> bool:
        """
        Save checkpoint with backup, validation, and disk space check.

        Args:
            checkpoint_data: Dictionary containing checkpoint data
            filename: Checkpoint filename
            experiment_name: Name of experiment (for logging)

        Returns:
            True if save successful, False otherwise
        """
        if filename is None:
            logging.error("save_checkpoint called with filename=None")
            return False

        ckpt_path = self.base_dir / str(filename)

        # Check disk space before saving
        if self._disk_guardian:
            if not self._disk_guardian.can_save_checkpoint(estimated_size_mb=500):
                logging.error("Insufficient disk space to save checkpoint")
                return False
        try:
            # Create backup if file exists
            if ckpt_path.exists():
                self._create_backup(ckpt_path, experiment_name)

            # Ensure rng states are included for reproducibility
            try:
                rng = {
                    'python_random_state': random.getstate(),
                    'numpy_random_state': np.random.get_state(),
                    'torch_cpu_rng_state': torch.get_rng_state()
                }
                if torch.cuda.is_available():
                    try:
                        rng['torch_cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
                    except (RuntimeError, AttributeError) as e:
                        rng['torch_cuda_rng_state_all'] = None
                        logging.warning(
                            "REPRODUCIBILITY: Failed to capture CUDA RNG state: %s. "
                            "Checkpoint may not be fully reproducible across GPU/CPU environments.",
                            e
                        )
                else:
                    rng['torch_cuda_rng_state_all'] = None
                    logging.info(
                        "CUDA not available - CPU-only RNG state captured. "
                        "Checkpoint reproducibility limited to CPU environments."
                    )
                checkpoint_data.setdefault('rng_states', rng)
            except (AttributeError, RuntimeError, ImportError) as e:
                logging.warning(
                    'CRITICAL: Could not capture RNG state for checkpoint: %s. '
                    'Reproducibility may be compromised.',
                    e
                )

            # Atomic save: write to temp file in same directory then replace
            tmp_path = ckpt_path.with_suffix('.tmp')
            try:
                # Attempt to use centralized version-safe saver writing to a path (more robust across torch versions)
                try:
                    # Save to temp file path (string) to allow torch to manage file handling
                    torch_save_safe(checkpoint_data, str(tmp_path), use_new_zipfile_serialization=True)
                except TypeError:
                    # Fallback: call without serialization flag
                    try:
                        torch_save_safe(checkpoint_data, str(tmp_path), use_new_zipfile_serialization=False)
                    except (RuntimeError, OSError):
                        # Final fallback: use direct torch.save to path
                        torch.save(checkpoint_data, str(tmp_path))

                # Ensure file is flushed to disk before atomic replace
                try:
                    with open(tmp_path, 'rb') as _f:
                        _f.flush()
                        os.fsync(_f.fileno())
                except OSError as e:
                    # Non-fatal: if fsync not possible, continue (atomic replace will still occur)
                    logging.debug("fsync failed (non-fatal): %s", e, exc_info=True)

                # Atomically replace
                os.replace(str(tmp_path), str(ckpt_path))

                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except (OSError, PermissionError):
                        pass

            except (OSError, RuntimeError, TypeError) as e:
                logging.error("Atomic save failed: %s", e, exc_info=True)
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except (OSError, PermissionError) as e2:
                    logging.debug("Failed to unlink temp checkpoint path during cleanup: %s", e2, exc_info=True)
                # Re-raise to be handled by outer except
                raise

            # Validate checkpoint
            if self._validate_checkpoint(ckpt_path, checkpoint_data):
                logging.info("Checkpoint saved: %s", ckpt_path)
                return True
            else:
                msg = f"Checkpoint validation failed: {ckpt_path}"
                logging.error(msg)
                if self.strict:
                    raise RuntimeError(msg)
                return False

        except (OSError, RuntimeError, ValueError) as e:
            logging.error("Failed to save checkpoint: %s", e)
            if self.strict:
                raise RuntimeError(f"Failed to save checkpoint: {e}") from e
            return False

    def load_checkpoint(
        self,
        filename: Optional[str],
        _experiment_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Load checkpoint with fallback to backup.

        Args:
            filename: Checkpoint filename
            _experiment_name: Name of experiment (for logging, unused)

        Returns:
            Checkpoint dictionary if successful, None otherwise
        """
        if filename is None:
            logging.error("load_checkpoint called with filename=None")
            return None

        ckpt_path = self.base_dir / str(filename)

        # Try primary checkpoint first
        if ckpt_path.exists():
            try:
                # Use centralized loader to handle versions that lack `weights_only` gracefully
                try:
                    checkpoint = torch_load_safe(ckpt_path, map_location='cpu', weights_only=False)
                except (TypeError, AttributeError) as e:
                    # Fallback to direct torch.load without weights_only
                    logging.warning(
                        "torch_load_safe failed or 'weights_only' unsupported; falling back to torch.load without weights_only: %s", e
                    )
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                logging.info("Loaded checkpoint: %s", ckpt_path)
                return checkpoint
            except (FileNotFoundError, OSError, RuntimeError) as e:
                logging.warning("Failed to load primary checkpoint: %s", e)

        # Try backup checkpoints
        for i in range(self.max_backups):
            backup_path = self.base_dir / f"{str(filename)}.backup_{i}"
            if backup_path.exists():
                try:
                    # Version-aware load with fallback
                    try:
                        try:
                            checkpoint = torch_load_safe(backup_path, map_location='cpu', weights_only=False)
                        except (TypeError, AttributeError) as e:
                            logging.warning("torch_load_safe failed for backup (weights_only unsupported): %s; falling back to torch.load without weights_only", e)
                            checkpoint = torch.load(backup_path, map_location='cpu')
                        logging.info("Loaded backup checkpoint: %s", backup_path)
                        return checkpoint
                    except (FileNotFoundError, OSError, RuntimeError) as e:
                        logging.debug("Failed to load backup %s using torch_load_safe: %s", backup_path, e, exc_info=True)
                except (FileNotFoundError, OSError, RuntimeError) as e:
                    logging.debug("Failed to load backup %d: %s", i, e)

        logging.debug(
            "No valid checkpoint found for %s (first run or checkpoint missing)",
            filename
        )
        return None

    def restore_rng_states(self, checkpoint: Dict):
        """Restore RNG states from a checkpoint if present.

        This will restore Python's random, NumPy RNG, PyTorch CPU RNG and
        CUDA RNG states (if available and stored). It is a no-op if RNG
        states are missing from the checkpoint.
        """
        rng = checkpoint.get('rng_states', {}) if isinstance(checkpoint, dict) else {}
        try:
            if 'python_random_state' in rng and rng['python_random_state'] is not None:
                random.setstate(rng['python_random_state'])
            if 'numpy_random_state' in rng and rng['numpy_random_state'] is not None:
                np.random.set_state(rng['numpy_random_state'])
            # Support multiple key names used historically across tests and code
            torch_cpu_state = None
            if 'torch_cpu_rng_state' in rng and rng['torch_cpu_rng_state'] is not None:
                torch_cpu_state = rng['torch_cpu_rng_state']
            elif 'torch_rng_state' in rng and rng['torch_rng_state'] is not None:
                torch_cpu_state = rng['torch_rng_state']

            if torch_cpu_state is not None:
                try:
                    torch.set_rng_state(torch_cpu_state)
                except (RuntimeError, ValueError, TypeError) as e:
                    logging.warning("Failed to restore CPU RNG state: %s", e)

            if 'torch_cuda_rng_state_all' in rng and rng['torch_cuda_rng_state_all'] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng['torch_cuda_rng_state_all'])
                except (RuntimeError, ValueError, TypeError) as e:
                    logging.warning("Failed to restore CUDA RNG state: %s", e)
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logging.warning("Failed to restore RNG states: %s", e)

    def _create_backup(self, ckpt_path: Path, _experiment_name: Optional[str]):
        """Create rolling backup - only if checkpoint exists.

        Locking protocol summary:
        - Attempt to create a lock file atomically using open(..., 'x') and write
          a unique token: "<pid>:<uuid>".
        - Wait up to ``self.backup_lock_timeout`` seconds for acquisition.
        - If a lock file exists and is older than ``self.stale_lock_seconds``
          it is considered stale and may be removed to allow progress.
        - On release, verify that the lock file contains the same token before
          unlinking to prevent accidental unlock by other processes.
        """
        if not ckpt_path.exists():
            return

        lock_file = self.base_dir / f"{ckpt_path.name}.backup.lock"

        try:
            # Remove stale lock if present and older than configured threshold
            try:
                if lock_file.exists():
                    age = time.time() - lock_file.stat().st_mtime
                    if age > (self.stale_lock_seconds + 1):
                        logging.warning("Removing stale backup lock (age %.0fs): %s", age, lock_file)
                        try:
                            lock_file.unlink()
                        except (OSError, PermissionError) as e:
                            logging.debug("Failed to unlink stale lock (%s): %s", lock_file, e)
                            # Best-effort fallback: try moving the stale lock aside to allow progress
                            try:
                                aside = lock_file.with_name(lock_file.name + f".stale.{uuid.uuid4()}")
                                lock_file.replace(aside)
                                logging.debug("Moved stale lock aside: %s -> %s", lock_file, aside)
                            except (OSError, PermissionError) as e2:
                                logging.debug("Failed to move stale lock aside: %s", e2)
            except OSError as e:
                logging.debug("Non-fatal error checking stale lock: %s", e, exc_info=True)

            # Try to acquire lock atomically using 'x' mode with timeout
            wait_time = 0.0
            acquired = False
            token = f"{os.getpid()}:{uuid.uuid4()}"
            while wait_time < float(self.backup_lock_timeout):
                try:
                    with open(lock_file, 'x') as lf:
                        lf.write(token)
                        lf.flush()
                        os.fsync(lf.fileno())
                    acquired = True
                    logging.debug("Acquired backup lock %s (token=%s)", lock_file, token)
                    logging.info("Acquired backup lock; proceeding to rotate/copy backups for %s", ckpt_path.name)
                    break
                except FileExistsError:
                    # Lock held by someone else; re-check for staleness
                    try:
                        age = time.time() - lock_file.stat().st_mtime
                        if age > (self.stale_lock_seconds + 1):
                            logging.warning("Lock appears stale (age %.0fs), removing: %s", age, lock_file)
                            try:
                                lock_file.unlink()
                                logging.debug("Removed stale lock: %s", lock_file)
                                # small backoff before trying to acquire
                                time.sleep(0.05)
                                wait_time += 0.05
                                continue
                            except (OSError, PermissionError) as e:
                                logging.debug("Failed to remove stale lock: %s", e)
                                # Best-effort fallback: try moving the stale lock aside so we can acquire
                                try:
                                    aside = lock_file.with_name(lock_file.name + f".stale.{uuid.uuid4()}")
                                    lock_file.replace(aside)
                                    logging.debug("Moved stale lock aside: %s -> %s", lock_file, aside)
                                except (OSError, PermissionError) as e2:
                                    logging.debug("Failed to move stale lock aside: %s", e2)
                    except OSError:
                        # If stat fails, treat as transient and retry
                        pass

                    # Lock is held and not stale; wait a bit and increment wait_time to respect configured timeout
                    time.sleep(0.05)
                    wait_time += 0.05

            if not acquired:
                logging.warning("Backup lock timeout for %s, could not acquire %s", ckpt_path.name, lock_file)
                return

            # We have acquired the lock. Perform rotation + copy within a try/finally so it will be released.
            try:
                # Roll backups: backup_n-1 -> backup_n
                try:
                    for i in range(self.max_backups - 1, 0, -1):
                        old_backup = self.base_dir / f"{ckpt_path.name}.backup_{i-1}"
                        new_backup = self.base_dir / f"{ckpt_path.name}.backup_{i}"
                        if old_backup.exists():
                            try:
                                if new_backup.exists():
                                    new_backup.unlink()
                                old_backup.rename(new_backup)
                            except (OSError, PermissionError) as e:
                                logging.warning("Failed to roll backup %d: %s", i, e)
                except (OSError, RuntimeError) as e:
                    logging.debug("Failed during backup rotation: %s", e, exc_info=True)

                # Copy current checkpoint to backup_0
                backup_0 = self.base_dir / f"{ckpt_path.name}.backup_0"
                try:
                    import shutil
                    logging.info("Creating backup_0 -> %s", backup_0)
                    shutil.copy2(str(ckpt_path), str(backup_0))
                    logging.info("Created backup: %s", backup_0)
                except (OSError, PermissionError) as e:
                    logging.warning("Failed to create backup_0: %s", e)
                except (shutil.Error, RuntimeError, TypeError) as e:
                    # Unexpected errors are logged explicitly; re-raise only if running in strict mode
                    logging.error("Unexpected error during backup copy: %s", e, exc_info=True)
                    if self.strict:
                        raise
                    return
            finally:
                # Release lock but only if the token still matches
                try:
                    try:
                        with open(lock_file, 'r') as lf:
                            existing = lf.read().strip()
                    except (OSError, FileNotFoundError):
                        existing = None
                    if existing == token:
                        try:
                            lock_file.unlink()
                            logging.debug("Released backup lock %s", lock_file)
                        except (OSError, PermissionError) as e:
                            logging.debug("Failed to unlink lock file on release: %s", e)
                    else:
                        logging.warning("Lock token mismatch on release: expected %s, found %s. Not unlinking %s", token, existing, lock_file)
                except (OSError, RuntimeError) as e:
                    logging.debug("Error during lock release: %s", e, exc_info=True)
        except (OSError, RuntimeError, PermissionError) as e:
            logging.warning("Backup creation failed: %s", e, exc_info=True)

    def _validate_checkpoint(self, ckpt_path: Path, original_data: Dict) -> bool:
        """
        Validate checkpoint by attempting to load it.

        Args:
            ckpt_path: Path to checkpoint file
            original_data: Original checkpoint data (for comparison)

        Returns:
            True if checkpoint is valid
        """
        try:
            # Version-aware load with fallback
            # Use version-aware loader
            try:
                from src.core.io_utils import torch_load_safe
                loaded = torch_load_safe(ckpt_path, map_location='cpu', weights_only=False)
            except (ImportError, AttributeError, TypeError) as e:
                # Fallback to direct torch.load without weights_only
                try:
                    loaded = torch.load(ckpt_path, map_location='cpu')
                except (FileNotFoundError, OSError, RuntimeError) as e:
                    logging.debug("Failed to load checkpoint during validation: %s", e)
                    return False
            # Basic validation: check that it's a dict and has some keys
            if not isinstance(loaded, dict):
                return False
            if len(loaded) == 0:
                return False
            return True
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logging.debug("Failed to validate checkpoint %s: %s", ckpt_path, e, exc_info=True)
            return False

