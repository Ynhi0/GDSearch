"""
Robust checkpoint management with backup, validation, and disk space awareness.

This module provides the RobustCheckpointManager class for production-grade
checkpoint handling with atomic writes, rollback, and integrity validation.
"""
import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


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

    def __init__(self, base_dir: str, max_backups: int = 3, min_free_gb: float = 1.0):
        """
        Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for checkpoints
            max_backups: Maximum number of backup files to keep
            min_free_gb: Minimum free disk space in GB
        """
        self.base_dir = Path(base_dir)
        self.max_backups = max_backups
        self.min_free_gb = min_free_gb
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
        filename: str,
        experiment_name: str
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
        ckpt_path = self.base_dir / filename
        
        # Check disk space before saving
        if self._disk_guardian:
            if not self._disk_guardian.can_save_checkpoint(estimated_size_mb=500):
                logging.error("Insufficient disk space to save checkpoint %s", filename)
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
                # Use binary write file handle to ensure fsync works
                with open(tmp_path, 'wb') as f:
                    # CRITICAL FIX: Version-aware save with fallback
                    try:
                        # Try new zipfile serialization (PyTorch >= 1.6)
                        torch.save(
                            checkpoint_data,
                            f,
                            _use_new_zipfile_serialization=True
                        )
                    except TypeError:
                        # Fallback for older PyTorch versions
                        logging.warning(
                            "_use_new_zipfile_serialization not supported, "
                            "using default serialization"
                        )
                        torch.save(checkpoint_data, f)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomically replace
                os.replace(str(tmp_path), str(ckpt_path))
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except (OSError, PermissionError):
                        pass

            # Validate checkpoint
            if self._validate_checkpoint(ckpt_path, checkpoint_data):
                logging.info("Checkpoint saved: %s", ckpt_path)
                return True
            else:
                logging.debug("Checkpoint validation failed: %s", ckpt_path)
                return False

        except (OSError, RuntimeError, ValueError) as e:
            logging.error("Failed to save checkpoint %s: %s", filename, e)
            return False

    def load_checkpoint(
        self,
        filename: str,
        _experiment_name: str = None
    ) -> Optional[Dict]:
        """
        Load checkpoint with fallback to backup.
        
        Args:
            filename: Checkpoint filename
            _experiment_name: Name of experiment (for logging, unused)
            
        Returns:
            Checkpoint dictionary if successful, None otherwise
        """
        ckpt_path = self.base_dir / filename

        # Try primary checkpoint first
        if ckpt_path.exists():
            try:
                # CRITICAL FIX: Version-aware load with fallback
                try:
                    checkpoint = torch.load(
                        ckpt_path,
                        map_location='cpu',
                        weights_only=False
                    )
                except TypeError:
                    # Fallback for PyTorch versions without weights_only parameter
                    logging.warning(
                        "weights_only parameter not supported, "
                        "using default torch.load behavior"
                    )
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                logging.info("Loaded checkpoint: %s", ckpt_path)
                return checkpoint
            except (FileNotFoundError, OSError, RuntimeError) as e:
                logging.warning("Failed to load primary checkpoint: %s", e)

        # Try backup checkpoints
        for i in range(self.max_backups):
            backup_path = self.base_dir / f"{filename}.backup_{i}"
            if backup_path.exists():
                try:
                    # Version-aware load with fallback
                    try:
                        checkpoint = torch.load(
                            backup_path,
                            map_location='cpu',
                            weights_only=False
                        )
                    except TypeError:
                        checkpoint = torch.load(backup_path, map_location='cpu')
                    logging.info("Loaded backup checkpoint: %s", backup_path)
                    return checkpoint
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
            if 'torch_cpu_rng_state' in rng and rng['torch_cpu_rng_state'] is not None:
                torch.set_rng_state(rng['torch_cpu_rng_state'])
            if 'torch_cuda_rng_state_all' in rng and rng['torch_cuda_rng_state_all'] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng['torch_cuda_rng_state_all'])
                except Exception as e:
                    logging.warning("Failed to restore CUDA RNG state: %s", e)
        except Exception as e:
            logging.warning("Failed to restore RNG states: %s", e)

    def _create_backup(self, ckpt_path: Path, _experiment_name: str):
        """Create rolling backup - only if checkpoint exists."""
        if not ckpt_path.exists():
            return
        
        # Create lock file for atomic backup operations
        lock_file = self.base_dir / f"{ckpt_path.name}.backup.lock"
        
        try:
            # Try to acquire lock (with timeout)
            max_wait = 30  # seconds
            wait_time = 0
            while lock_file.exists() and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1
            
            if wait_time >= max_wait:
                logging.warning("Backup lock timeout for %s", ckpt_path.name)
                return
            
            # Acquire lock
            lock_file.touch()
            
            try:
                # Roll backups: backup_2 -> backup_3, backup_1 -> backup_2, etc.
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
                
                # Copy current checkpoint to backup_0
                backup_0 = self.base_dir / f"{ckpt_path.name}.backup_0"
                try:
                    import shutil
                    shutil.copy2(str(ckpt_path), str(backup_0))
                except (OSError, PermissionError) as e:
                    logging.warning("Failed to create backup_0: %s", e)
            finally:
                # Release lock
                try:
                    lock_file.unlink()
                except (OSError, PermissionError):
                    pass
                    
        except Exception as e:
            logging.warning("Backup creation failed: %s", e)

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
            try:
                loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            except TypeError:
                loaded = torch.load(ckpt_path, map_location='cpu')
            # Basic validation: check that it's a dict and has some keys
            if not isinstance(loaded, dict):
                return False
            if len(loaded) == 0:
                return False
            return True
        except Exception:
            return False
