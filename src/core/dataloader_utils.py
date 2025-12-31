"""
DataLoader utilities for creating reproducible, deterministic data loaders.

This module provides a centralized implementation of make_dataloader that ensures
consistent behavior across all experiments.
"""
import functools
import random
from typing import Optional, Callable, Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _worker_init(worker_id: int, seed: int):
    """Initialize worker with deterministic seed."""
    worker_seed = int(seed) + worker_id + 1
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        torch.manual_seed(worker_seed)
    except Exception as e:
        import logging
        logging.debug("Could not set torch.manual_seed in worker %s: %s", worker_id, e, exc_info=True)


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Optional[Callable] = None,
    sampler: Optional[Sampler] = None,
    drop_last: bool = False,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Create a DataLoader with deterministic worker seeding when `seed` is provided.
    
    - If `seed` is not None, a `torch.Generator` is created and `worker_init_fn` 
      seeds python, numpy and torch RNGs for each worker deterministically.
    - If `sampler` is provided, it will be used and `shuffle` will be ignored.
    - `persistent_workers` requires PyTorch >= 1.7.0 and num_workers > 0
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data (ignored if sampler provided)
        seed: Random seed for reproducibility
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        collate_fn: Custom collate function
        sampler: Custom sampler (overrides shuffle)
        drop_last: Drop last incomplete batch
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        DataLoader with configured settings
        
    WINDOWS COMPATIBILITY: On Windows, num_workers is forced to 0 to prevent
    multiprocessing issues. This ensures testing works on Windows while still
    allowing full multiprocessing on Kaggle/Linux.
    """
    # AUDIT FIX: Force num_workers=0 on Windows to prevent hanging/crashes
    import platform
    import logging
    if platform.system() == 'Windows' and num_workers > 0:
        logging.debug(f"Windows detected: forcing num_workers=0 (was {num_workers}) for stability")
        num_workers = 0
        # persistent_workers requires num_workers > 0, so disable it
        persistent_workers = False
    
    generator = None
    worker_init_fn = None

    if seed is not None:
        try:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            worker_init_fn = functools.partial(_worker_init, seed=seed)
        except Exception as e:
            logging.debug("Failed to create deterministic generator/worker_init_fn: %s", e, exc_info=True)
            generator = None
            worker_init_fn = None

    dl_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    if collate_fn is not None:
        dl_kwargs['collate_fn'] = collate_fn

    if sampler is not None:
        dl_kwargs['sampler'] = sampler

    if worker_init_fn is not None:
        dl_kwargs['worker_init_fn'] = worker_init_fn

    if generator is not None and sampler is None:
        dl_kwargs['generator'] = generator

    # Add persistent_workers only if PyTorch supports it and num_workers > 0
    if persistent_workers and num_workers > 0:
        try:
            pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            if pytorch_version >= (1, 7):
                dl_kwargs['persistent_workers'] = True
        except Exception as e:
            logging.debug("Failed to parse PyTorch version for persistent_workers: %s", e, exc_info=True)  # Skip if version parsing fails

    loader = DataLoader(dataset, **dl_kwargs)
    
    # CRITICAL: Add basic metadata for test-leakage prevention
    # Callers should override with more specific values if available
    if not hasattr(loader, 'name'):
        loader.name = 'unknown'
    if not hasattr(loader, '_split_type'):
        loader._split_type = 'unknown'
    if not hasattr(loader, '_dataset_uid'):
        # Use helper to compute length for unknown dataset types (e.g., HuggingFace datasets)
        from src.utils.safe_len import len_sized
        try:
            loader._dataset_uid = f'dataset_{len_sized(dataset)}'
        except Exception:
            loader._dataset_uid = 'dataset_unknown'
    
    return loader
