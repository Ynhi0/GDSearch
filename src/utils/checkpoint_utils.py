"""
Checkpoint utilities for atomic saves, versioning, and robust resume.

This module provides best-practice checkpoint handling patterns including:
- Atomic saves with temp file + fsync + rename
- Comprehensive metadata (config, git commit, timestamps, RNG states)
- Robust loading with validation
- Resume support with progress tracking

M2: Implementation of best-practice checkpoint/resume logic
"""
import os
import tempfile
import logging
import datetime
import subprocess
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch


def save_checkpoint_atomic(checkpoint_data: Dict[str, Any], checkpoint_path: Path) -> None:
    """
    Save checkpoint atomically to prevent corruption.
    
    Uses temp file + atomic rename pattern:
    1. Save to temporary file in same directory
    2. Sync to disk with fsync
    3. Atomic rename to final path
    
    This ensures that checkpoint files are never left in a corrupted state
    even if the process is interrupted during save.
    
    Args:
        checkpoint_data: Dictionary with model_state_dict, optimizer_state_dict, epoch, etc.
        checkpoint_path: Final checkpoint path
        
    Raises:
        RuntimeError: If save fails
    """
    # Create temp file in same directory (ensures same filesystem for atomic rename)
    temp_dir = checkpoint_path.parent
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=temp_dir,
        delete=False,
        prefix='.tmp_checkpoint_',
        suffix='.pt'
    ) as tmp_file:
        temp_path = Path(tmp_file.name)
        
        try:
            # Save to temp file
            torch.save(checkpoint_data, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())  # Force write to disk
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e
    
    try:
        # Atomic rename with Windows compatibility
        if os.name == 'nt':  # Windows
            # On Windows, use MoveFileEx with REPLACE_EXISTING flag for atomicity
            import ctypes
            try:
                # MOVEFILE_REPLACE_EXISTING = 0x1
                result = ctypes.windll.kernel32.MoveFileExW(
                    str(temp_path),
                    str(checkpoint_path),
                    0x1  # MOVEFILE_REPLACE_EXISTING
                )
                if not result:
                    raise OSError(f"MoveFileEx failed with error code: {ctypes.GetLastError()}")
            except Exception as e:
                raise RuntimeError(f"Atomic rename failed on Windows: {e}") from e
        else:  # Unix/Linux
            # On Unix, replace() uses rename(2) which is atomic
            temp_path.replace(checkpoint_path)
        logging.info(f"Checkpoint saved atomically: {checkpoint_path}")
    except Exception as e:
        # Clean up temp file on rename error
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to atomically rename checkpoint: {e}") from e


def create_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    additional_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create checkpoint with comprehensive metadata.
    
    Includes:
    - Model and optimizer state
    - Training progress (epoch, best metric)
    - Configuration for reproducibility
    - Timestamp and version info
    - Git commit hash (if available)
    - Random states for full reproducibility
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        best_metric: Best validation metric so far
        config: Experiment configuration dictionary
        additional_state: Optional additional state to save
        
    Returns:
        Complete checkpoint dictionary ready for saving
    """
    checkpoint = {
        # Core training state
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        
        # Configuration
        'config': config,
        
        # Metadata
        'timestamp': datetime.datetime.now().isoformat(),
        'checkpoint_version': '2.0',
        'pytorch_version': torch.__version__,
        
        # Reproducibility
        'random_states': {
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate()
        }
    }
    
    # Try to get git commit hash
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        checkpoint['git_commit'] = git_hash
    except Exception:
        checkpoint['git_commit'] = None
    
    # Add any additional state
    if additional_state:
        checkpoint['additional_state'] = additional_state
    
    return checkpoint


def load_checkpoint_safe(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load checkpoint with validation and error handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to map checkpoint to
        strict: If True, enforce exact state dict match
        
    Returns:
        Checkpoint metadata (epoch, best_metric, config, etc.)
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is corrupted or incompatible
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        # Load checkpoint - set weights_only=False for full checkpoint with metadata
        try:
            # PyTorch 2.6+ requires explicit weights_only parameter
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Validate checkpoint structure
        required_keys = ['epoch', 'model_state_dict']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise RuntimeError(f"Checkpoint missing required keys: {missing_keys}")
        
        # Load model state
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logging.info(f"Model state loaded (strict={strict})")
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to load model state (strict mode): {e}") from e
            else:
                logging.warning(f"Partial model state load (non-strict): {e}")
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("Optimizer state loaded")
            except Exception as e:
                logging.warning(f"Failed to load optimizer state: {e}")
        
        # Restore random states for reproducibility
        if 'random_states' in checkpoint:
            try:
                torch.set_rng_state(checkpoint['random_states']['torch'])
                if torch.cuda.is_available() and checkpoint['random_states']['torch_cuda']:
                    torch.cuda.set_rng_state_all(checkpoint['random_states']['torch_cuda'])
                np.random.set_state(checkpoint['random_states']['numpy'])
                random.setstate(checkpoint['random_states']['python'])
                logging.info("Random states restored")
            except Exception as e:
                logging.warning(f"Failed to restore random states: {e}")
        
        # Return metadata
        metadata = {
            'epoch': checkpoint['epoch'],
            'best_metric': checkpoint.get('best_metric'),
            'config': checkpoint.get('config'),
            'timestamp': checkpoint.get('timestamp'),
            'git_commit': checkpoint.get('git_commit'),
            'additional_state': checkpoint.get('additional_state')
        }
        
        logging.info(f"Checkpoint loaded: epoch {metadata['epoch']}, best_metric {metadata['best_metric']}")
        return metadata
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}") from e


class CheckpointManager:
    """
    Manages checkpoint saving with automatic cleanup.
    
    Features:
    - Keep last N checkpoints
    - Keep best K checkpoints by metric
    - Keep milestone checkpoints (every M epochs)
    - Atomic saves
    - Automatic cleanup of old checkpoints
    
    Example:
        manager = CheckpointManager(
            checkpoint_dir=Path('checkpoints'),
            keep_last=3,
            keep_best=3,
            keep_milestones=[10, 25, 50],
            metric_mode='max'
        )
        
        # During training:
        checkpoint = create_checkpoint(model, optimizer, epoch, val_acc, config)
        manager.save_checkpoint(checkpoint, epoch, val_acc, is_best=(val_acc > best_acc))
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last: int = 3,
        keep_best: int = 3,
        keep_milestones: Optional[List[int]] = None,
        metric_mode: str = 'max'  # 'max' for accuracy, 'min' for loss
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last: Number of most recent checkpoints to keep
            keep_best: Number of best checkpoints (by metric) to keep
            keep_milestones: List of epoch numbers to always keep
            metric_mode: 'max' (higher is better) or 'min' (lower is better)
        """
        if keep_last < 0 or keep_best < 0:
            raise ValueError(f"keep_last and keep_best must be >= 0, got keep_last={keep_last}, keep_best={keep_best}")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.keep_milestones = keep_milestones or []
        self.metric_mode = metric_mode
        
        self.checkpoints: List[Tuple[int, float, Path]] = []  # List of (epoch, metric, path)
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self) -> None:
        """Scan checkpoint directory and load existing checkpoint metadata."""
        for ckpt_file in self.checkpoint_dir.glob('*.pt'):
            try:
                # Try to extract epoch and metric from filename
                # Format: checkpoint_epoch{N}_metric{M}.pt
                parts = ckpt_file.stem.split('_')
                epoch = None
                metric = None
                for part in parts:
                    if part.startswith('epoch'):
                        epoch = int(part.replace('epoch', ''))
                    elif part.startswith('metric'):
                        metric = float(part.replace('metric', ''))
                
                if epoch is not None:
                    # If metric not in filename, try loading checkpoint
                    if metric is None:
                        try:
                            ckpt = torch.load(ckpt_file, map_location='cpu')
                            metric = ckpt.get('best_metric', 0.0)
                        except Exception:
                            metric = 0.0
                    
                    self.checkpoints.append((epoch, metric, ckpt_file))
            except Exception as e:
                logging.warning(f"Could not parse checkpoint {ckpt_file}: {e}")
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        metric: float,
        is_best: bool = False
    ) -> Path:
        """
        Save checkpoint and manage cleanup.
        
        Args:
            checkpoint_data: Checkpoint dictionary from create_checkpoint()
            epoch: Current epoch number
            metric: Current validation metric value
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint filename
        if is_best:
            filename = f'best_checkpoint_epoch{epoch}_metric{metric:.4f}.pt'
        elif epoch in self.keep_milestones:
            filename = f'milestone_checkpoint_epoch{epoch}.pt'
        else:
            filename = f'checkpoint_epoch{epoch}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save atomically
        save_checkpoint_atomic(checkpoint_data, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append((epoch, metric, checkpoint_path))
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints based on retention policy."""
        # Sort by epoch (most recent first)
        sorted_by_epoch = sorted(self.checkpoints, key=lambda x: x[0], reverse=True)
        
        # Sort by metric (best first)
        if self.metric_mode == 'max':
            sorted_by_metric = sorted(self.checkpoints, key=lambda x: x[1], reverse=True)
        else:
            sorted_by_metric = sorted(self.checkpoints, key=lambda x: x[1])
        
        # Determine which to keep
        keep_paths = set()
        
        # Keep last N
        for epoch, metric, path in sorted_by_epoch[:self.keep_last]:
            keep_paths.add(path)
        
        # Keep best K
        for epoch, metric, path in sorted_by_metric[:self.keep_best]:
            keep_paths.add(path)
        
        # Keep milestones and specially named checkpoints
        for epoch, metric, path in self.checkpoints:
            if epoch in self.keep_milestones:
                keep_paths.add(path)
            if 'best_' in path.name or 'milestone_' in path.name:
                keep_paths.add(path)
        
        # Delete checkpoints not in keep set
        for epoch, metric, path in self.checkpoints:
            if path not in keep_paths and path.exists():
                logging.info(f"Removing old checkpoint: {path.name}")
                path.unlink()
        
        # Update tracking list
        self.checkpoints = [(e, m, p) for e, m, p in self.checkpoints if p in keep_paths]
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        if not self.checkpoints:
            return None
        sorted_ckpts = sorted(self.checkpoints, key=lambda x: x[0], reverse=True)
        return sorted_ckpts[0][2]
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint by metric."""
        if not self.checkpoints:
            return None
        if self.metric_mode == 'max':
            sorted_ckpts = sorted(self.checkpoints, key=lambda x: x[1], reverse=True)
        else:
            sorted_ckpts = sorted(self.checkpoints, key=lambda x: x[1])
        return sorted_ckpts[0][2]
