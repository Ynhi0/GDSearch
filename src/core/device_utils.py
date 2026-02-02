"""
Safe device handling utilities for CPU/GPU operations.

Provides defensive wrappers to prevent device mismatch errors,
handle GPU OOM during tensor transfers, and enable graceful fallbacks.
"""

import torch
import logging
from typing import Union, Optional


def safe_to_device(
    tensor: torch.Tensor,
    device: Union[str, torch.device],
    error_context: str = ""
) -> torch.Tensor:
    """
    Move tensor to device with proper error handling and OOM protection.
    
    This function prevents common device-related errors:
    1. GPU OOM during .to(device) - falls back to CPU
    2. Invalid device index (e.g., cuda:5 when only 2 GPUs)
    3. Device type mismatch (tensor already on correct device)
    
    Args:
        tensor: Input tensor to move
        device: Target device (string or torch.device)
        error_context: Context string for error messages (e.g., "batch 5, epoch 2")
    
    Returns:
        Tensor on target device (or CPU if GPU OOM)
    
    Raises:
        ValueError: If device is invalid
        RuntimeError: If transfer fails for non-OOM reasons
    
    Example:
        >>> data = torch.randn(1000, 784)
        >>> data = safe_to_device(data, "cuda:0", error_context="training batch 5")
    """
    # Normalize device to torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    # Early return if already on correct device
    try:
        if tensor.device == device:
            return tensor
    except Exception as e:
        logging.debug(f"Could not check tensor device: {e}")
    
    # Attempt transfer with error handling
    try:
        return tensor.to(device)
    
    except RuntimeError as e:
        error_str = str(e).lower()
        
        # Handle GPU OOM
        if "out of memory" in error_str or "oom" in error_str:
            context_msg = f" ({error_context})" if error_context else ""
            logging.error(
                f"GPU OOM during tensor transfer{context_msg}. "
                f"Falling back to CPU. Original error: {e}"
            )
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fall back to CPU
            cpu_device = torch.device("cpu")
            logging.warning(f"Transferred tensor to CPU instead of {device}")
            return tensor.to(cpu_device)
        
        # Handle invalid device
        elif "device" in error_str or "cuda" in error_str:
            context_msg = f" ({error_context})" if error_context else ""
            logging.error(
                f"Invalid device {device}{context_msg}. "
                f"Check available GPUs with torch.cuda.device_count(). "
                f"Original error: {e}"
            )
            raise ValueError(
                f"Device {device} is not available. "
                f"Available devices: CPU, cuda:0 to cuda:{torch.cuda.device_count()-1 if torch.cuda.is_available() else 'none'}"
            ) from e
        
        else:
            # Unknown CUDA error - re-raise
            context_msg = f" ({error_context})" if error_context else ""
            logging.error(f"Unexpected error during tensor transfer{context_msg}: {e}")
            raise


def get_available_device(prefer_gpu: bool = True, gpu_index: int = 0) -> torch.device:
    """
    Get best available device with validation.
    
    Args:
        prefer_gpu: If True, use GPU when available; if False, always use CPU
        gpu_index: GPU index to use (default: 0)
    
    Returns:
        torch.device object (validated to be available)
    
    Example:
        >>> device = get_available_device(prefer_gpu=True, gpu_index=0)
        >>> print(device)  # cuda:0 or cpu
    """
    if not prefer_gpu:
        return torch.device("cpu")
    
    if not torch.cuda.is_available():
        logging.info("GPU requested but not available, using CPU")
        return torch.device("cpu")
    
    # Validate GPU index
    num_gpus = torch.cuda.device_count()
    if gpu_index >= num_gpus:
        logging.warning(
            f"GPU {gpu_index} requested but only {num_gpus} GPU(s) available. "
            f"Using GPU 0 instead."
        )
        gpu_index = 0
    
    device = torch.device(f"cuda:{gpu_index}")
    
    # Verify device is actually usable
    try:
        # Try to allocate a small tensor to verify device works
        test_tensor = torch.zeros(1, device=device)
        del test_tensor
        logging.info(f"Using device: {device}")
        return device
    except RuntimeError as e:
        logging.error(f"Device {device} exists but is not usable: {e}. Falling back to CPU.")
        return torch.device("cpu")


def validate_device_compatibility(
    model: torch.nn.Module,
    data: torch.Tensor,
    target_device: torch.device
) -> None:
    """
    Verify model and data are on compatible devices.
    
    Raises:
        RuntimeError: If devices are incompatible
    
    Example:
        >>> model = SimpleMLP(784, 128, 10).to("cuda")
        >>> data = torch.randn(32, 784).to("cpu")
        >>> validate_device_compatibility(model, data, torch.device("cuda"))
        RuntimeError: Device mismatch detected!
    """
    # Get model device (check first parameter)
    model_device = next(model.parameters()).device
    data_device = data.device
    
    if model_device != target_device:
        raise RuntimeError(
            f"Model is on {model_device} but target device is {target_device}. "
            f"Call model.to('{target_device}') before training."
        )
    
    if data_device != target_device:
        raise RuntimeError(
            f"Data is on {data_device} but target device is {target_device}. "
            f"Ensure data is transferred to device before forward pass."
        )
    
    logging.debug(f"Device compatibility OK: model={model_device}, data={data_device}, target={target_device}")


def safe_model_init(
    model_class,
    *args,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> tuple[torch.nn.Module, torch.device]:
    """
    Initialize model with OOM protection.
    
    If GPU OOM occurs during model initialization, automatically falls back to CPU.
    
    Args:
        model_class: Model class to instantiate
        *args: Positional arguments for model constructor
        device: Target device (None = auto-select)
        **kwargs: Keyword arguments for model constructor
    
    Returns:
        Tuple of (model, actual_device)
        - model: Initialized model
        - actual_device: Device model ended up on (may differ from requested if OOM)
    
    Example:
        >>> model, device = safe_model_init(SimpleMLP, 784, 128, 10, device="cuda")
        >>> # If GPU OOM, returns (model_on_cpu, torch.device("cpu"))
    """
    # Determine target device
    if device is None:
        device = get_available_device(prefer_gpu=True)
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Try to initialize on target device
    try:
        model = model_class(*args, **kwargs)
        model = model.to(device)
        logging.info(f"Model initialized successfully on {device}")
        return model, device
    
    except RuntimeError as e:
        error_str = str(e).lower()
        
        # Handle GPU OOM during initialization
        if "out of memory" in error_str or "oom" in error_str:
            logging.warning(
                f"GPU OOM during model initialization on {device}. "
                f"Falling back to CPU. Consider using a smaller model. "
                f"Original error: {e}"
            )
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Retry on CPU
            cpu_device = torch.device("cpu")
            model = model_class(*args, **kwargs)
            model = model.to(cpu_device)
            logging.info(f"Model initialized successfully on CPU (fallback)")
            return model, cpu_device
        
        else:
            # Non-OOM error - re-raise
            logging.error(f"Model initialization failed: {e}")
            raise


def check_gpu_memory(device: Union[str, torch.device], required_mb: float = 100) -> bool:
    """
    Check if GPU has sufficient memory available.
    
    Args:
        device: GPU device to check
        required_mb: Required memory in MB
    
    Returns:
        True if sufficient memory available, False otherwise
    
    Example:
        >>> if check_gpu_memory("cuda:0", required_mb=500):
        ...     # Proceed with GPU training
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type != "cuda":
        return True  # CPU always has "enough" memory (OS will swap)
    
    if not torch.cuda.is_available():
        return False
    
    try:
        torch.cuda.set_device(device)
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        
        if free_mb < required_mb:
            logging.warning(
                f"GPU {device} has only {free_mb:.0f} MB free, "
                f"{required_mb:.0f} MB required"
            )
            return False
        
        logging.debug(f"GPU {device} has {free_mb:.0f} MB free (sufficient)")
        return True
    
    except RuntimeError as e:
        logging.error(f"Could not check GPU memory: {e}")
        return False


def clear_gpu_memory(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Clear GPU cache and synchronize.
    
    Should be called:
    - After exceptions during training
    - Between experiments
    - When memory usage is high
    
    Args:
        device: Device to clear (None = all GPUs)
    
    Example:
        >>> try:
        ...     # Training code
        ... except RuntimeError as e:
        ...     clear_gpu_memory()
        ...     raise
    """
    if not torch.cuda.is_available():
        return
    
    try:
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if device.type == "cuda":
                torch.cuda.set_device(device)
        
        torch.cuda.synchronize()  # Wait for all kernels to finish
        torch.cuda.empty_cache()  # Release cached memory
        
        logging.debug("GPU memory cleared")
    
    except Exception as e:
        logging.debug(f"Could not clear GPU memory: {e}")
