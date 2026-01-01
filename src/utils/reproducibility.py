"""
GPU Reproducibility and Determinism Utilities.

CRITICAL FIX: Addresses non-deterministic GPU behavior that causes
irreproducible results even with torch.manual_seed(). This module provides
comprehensive determinism enforcement for scientific reproducibility.

Key Issues Addressed:
1. CUDA kernel non-determinism (cudnn.benchmark, convolution algorithms)
2. Non-deterministic atomic operations
3. Random number generator state across CPU/GPU
"""

import torch
import numpy as np
import random
import os
import logging
from typing import Optional
import warnings


def set_reproducibility_mode(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    warn_nondeterministic_ops: bool = True
) -> None:
    """
    Configure PyTorch for full reproducibility across CPU and GPU.
    
    CRITICAL for academic research: Without this, repeated runs with the
    same seed will produce DIFFERENT results on GPU, making experiments
    irreproducible and scientifically invalid.
    
    Args:
        seed: Random seed for all RNGs
        deterministic: If True, use deterministic algorithms (slower but reproducible)
        benchmark: If True, enable cudnn.benchmark (faster but non-deterministic)
        warn_nondeterministic_ops: If True, warn when non-deterministic ops are used
        
    Side Effects:
        - Sets random seeds for Python, NumPy, PyTorch
        - Configures CUDA determinism
        - May reduce performance for reproducibility
    """
    # Set seeds for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Configure CuDNN for determinism
    if torch.backends.cudnn.is_available():
        # Disable benchmark mode: prevents auto-tuning of convolution algorithms
        # which introduces non-determinism
        torch.backends.cudnn.benchmark = benchmark
        
        # Enable deterministic mode: forces deterministic algorithms
        torch.backends.cudnn.deterministic = deterministic
        
        if deterministic and benchmark:
            warnings.warn(
                "Both deterministic=True and benchmark=True. "
                "This may still produce non-deterministic results. "
                "For strict reproducibility, set benchmark=False."
            )
    
    # PyTorch 1.8+: Use deterministic algorithms globally
    try:
        torch.use_deterministic_algorithms(deterministic)
    except AttributeError:
        # Older PyTorch versions
        logging.debug("torch.use_deterministic_algorithms() not available in this PyTorch version")
    except RuntimeError as e:
        # Some operations don't have deterministic implementations
        logging.warning(f"Could not enable deterministic algorithms: {e}")
    
    # Configure CUBLAS workspace for deterministic operations
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Optional: Warn about non-deterministic operations
    if warn_nondeterministic_ops and deterministic:
        try:
            torch.set_warn_always(True)
        except AttributeError:
            pass  # Not available in all PyTorch versions
    
    logging.info(
        f"Reproducibility mode configured: seed={seed}, "
        f"deterministic={deterministic}, benchmark={benchmark}"
    )


def check_reproducibility_status() -> dict:
    """
    Check current reproducibility configuration.
    
    Returns:
        Dict with configuration status
    """
    status = {
        'cuda_available': torch.cuda.is_available(),
        'cudnn_available': torch.backends.cudnn.is_available(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic if torch.backends.cudnn.is_available() else None,
        'cudnn_benchmark': torch.backends.cudnn.benchmark if torch.backends.cudnn.is_available() else None,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set'),
    }
    
    try:
        status['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    except AttributeError:
        status['deterministic_algorithms'] = 'Not available (PyTorch < 1.8)'
    
    return status


def warn_if_nondeterministic() -> None:
    """
    Issue warnings if reproducibility is not properly configured.
    
    Call this at the start of experiments to catch configuration issues.
    """
    status = check_reproducibility_status()
    
    issues = []
    
    if status['cuda_available']:
        if not status['cudnn_deterministic']:
            issues.append("cudnn.deterministic=False: GPU operations may be non-deterministic")
        
        if status['cudnn_benchmark']:
            issues.append("cudnn.benchmark=True: Convolution algorithms may vary between runs")
        
        if status['cublas_workspace_config'] == 'Not set':
            issues.append("CUBLAS_WORKSPACE_CONFIG not set: Some operations may be non-deterministic")
        
        if status['deterministic_algorithms'] == False:
            issues.append("Deterministic algorithms disabled: Results may not be reproducible")
    
    if issues:
        warning_msg = (
            "⚠️ REPRODUCIBILITY WARNING: Non-deterministic configuration detected:\n" +
            "\n".join(f"  - {issue}" for issue in issues) +
            "\n\nFor reproducible results, call set_reproducibility_mode(seed, deterministic=True, benchmark=False)"
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        logging.warning(warning_msg)
    else:
        logging.info("✅ Reproducibility configuration: All checks passed")


def get_rng_state() -> dict:
    """
    Get current state of all random number generators.
    
    Useful for saving/restoring RNG state for checkpointing.
    
    Returns:
        Dict containing RNG states
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()
        state['torch_cuda_all'] = torch.cuda.get_rng_state_all()
    
    return state


def set_rng_state(state: dict) -> None:
    """
    Restore random number generator states from saved state.
    
    Args:
        state: Dict from get_rng_state()
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if torch.cuda.is_available():
        if 'torch_cuda' in state:
            torch.cuda.set_rng_state(state['torch_cuda'])
        if 'torch_cuda_all' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda_all'])


# Convenience function for experiment scripts
def setup_experiment_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    check_config: bool = True
) -> None:
    """
    One-line setup for reproducible experiments.
    
    Call this at the start of every experiment script.
    
    Args:
        seed: Random seed
        deterministic: Enable deterministic algorithms
        check_config: Issue warnings if config is suboptimal
    """
    set_reproducibility_mode(seed, deterministic=deterministic, benchmark=False)
    
    if check_config:
        warn_if_nondeterministic()
    
    logging.info(f"Experiment reproducibility initialized with seed={seed}")
