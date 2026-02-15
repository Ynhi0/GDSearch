"""
Device transfer safety utilities for PyTorch models and tensors.

Provides safe .to(device) operations with OOM handling and fallback.
"""

import logging
from typing import Union, TypeVar
import torch

T = TypeVar('T', torch.nn.Module, torch.Tensor)


def safe_device_transfer(
    obj: T,
    device: torch.device,
    operation: str = "device transfer",
    fallback_to_cpu: bool = True
) -> T:
    """
    Safely transfer model or tensor to device with OOM handling.
    
    Args:
        obj: PyTorch model or tensor to transfer
        device: Target device
        operation: Description for error messages
        fallback_to_cpu: If True, fall back to CPU on GPU OOM
    
    Returns:
        Object on target device (or CPU if fallback occurred)
    
    Example:
        >>> model = SimpleMLP()
        >>> model = safe_device_transfer(model, device, "model initialization")
    """
    try:
        return obj.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and fallback_to_cpu:
            logging.warning(
                f"GPU OOM during {operation}, falling back to CPU. "
                f"Original error: {e}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return obj.to('cpu')
        else:
            logging.error(f"Device transfer failed during {operation}: {e}")
            raise


# Re-export gpu_safe_operation from error_handling_patterns
from src.utils.error_handling_patterns import gpu_safe_operation

__all__ = ['safe_device_transfer', 'gpu_safe_operation']
