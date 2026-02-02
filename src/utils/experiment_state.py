"""
Experiment State Management Utilities

Provides functions to reset global state between experiments to prevent state bleeding.

Bug Fix: Ensures each experiment starts with clean RNG state and cleared GPU memory.
"""

import logging
import random
import torch
import numpy as np
from typing import Optional


def reset_experiment_state(seed: int, device: Optional[torch.device] = None) -> None:
    """
    Reset all global state before running an experiment.
    
    Prevents state bleeding between experiments by:
    - Resetting all RNG states (Python, NumPy, PyTorch)
    - Clearing GPU cache if available
    - Resetting CUDNN settings to defaults
    
    Bug Fix: Ensures experiments are independent even when run sequentially.
    
    Args:
        seed: Random seed for reproducibility
        device: Device to clear cache on (auto-detected if None)
        
    Example:
        >>> for config in experiment_configs:
        ...     reset_experiment_state(config['seed'])
        ...     run_experiment(config)
    """
    # Reset all RNG states
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Clear GPU cache to prevent memory leaks between experiments
        torch.cuda.empty_cache()
        
        # Reset CUDNN settings to defaults
        # Note: These should match the deterministic settings if reproducibility is needed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        
        # Reset peak memory stats for accurate tracking
        torch.cuda.reset_peak_memory_stats()
        
        logging.debug(
            f"Reset CUDA state: seed={seed}, "
            f"memory_allocated={torch.cuda.memory_allocated() / 1024**2:.1f}MB"
        )
    
    logging.debug(f"Reset experiment state with seed={seed}")


def enable_deterministic_mode(seed: int) -> None:
    """
    Enable fully deterministic mode for reproducibility.
    
    Sets all RNG seeds and enables deterministic CUDNN operations.
    Note: This may reduce performance but ensures reproducibility.
    
    Args:
        seed: Random seed for all RNGs
    """
    reset_experiment_state(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # PyTorch 1.8+ has additional deterministic settings
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            logging.warning(f"Could not enable deterministic algorithms: {e}")
    
    logging.info(f"Enabled deterministic mode with seed={seed}")


def get_gpu_memory_status() -> dict:
    """
    Get current GPU memory status.
    
    Returns:
        Dictionary with memory statistics or empty dict if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
    }


def clear_gpu_memory() -> None:
    """
    Aggressively clear GPU memory.
    
    Useful between experiments to prevent OOM errors.
    """
    if not torch.cuda.is_available():
        return
    
    # Multiple rounds of cache clearing can help in some cases
    for _ in range(3):
        torch.cuda.empty_cache()
    
    # Synchronize to ensure operations complete
    torch.cuda.synchronize()
    
    mem_status = get_gpu_memory_status()
    logging.debug(
        f"Cleared GPU memory: allocated={mem_status.get('allocated_mb', 0):.1f}MB, "
        f"reserved={mem_status.get('reserved_mb', 0):.1f}MB"
    )


def validate_experiment_config(config: dict) -> None:
    """
    Validate experiment configuration has required fields.
    
    Args:
        config: Experiment configuration dictionary
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ['seed']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Experiment config missing required field: '{field}'. "
                f"Config must contain: {required_fields}"
            )
    
    # Validate seed is a valid integer
    try:
        seed = int(config['seed'])
        if seed < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}")
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid seed value: {config.get('seed')}. Must be non-negative integer."
        ) from e


def safe_experiment_wrapper(run_func):
    """
    Decorator to wrap experiment functions with state reset.
    
    Ensures state is reset before each experiment and GPU memory is cleared after.
    
    Example:
        @safe_experiment_wrapper
        def run_experiment(config):
            # Your experiment code
            return results
    """
    import functools
    
    @functools.wraps(run_func)
    def wrapper(config, *args, **kwargs):
        # Validate config has required fields
        validate_experiment_config(config)
        
        # Reset state before experiment
        reset_experiment_state(config['seed'])
        
        try:
            # Run experiment
            result = run_func(config, *args, **kwargs)
            return result
        finally:
            # Clear GPU memory after experiment (even if it fails)
            clear_gpu_memory()
    
    return wrapper
