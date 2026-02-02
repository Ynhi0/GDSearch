"""
Reusable error handling patterns for robust experiment code.

This module provides utilities for common error handling scenarios:
- GPU OOM handling with cleanup
- Resource cleanup with try/finally
- Informative error messages with context
- Logging before re-raising exceptions
- PyTorch-specific error handling

Example:
    from src.utils.error_handling_patterns import gpu_safe_operation, log_and_reraise
    
    with gpu_safe_operation("Training epoch"):
        model(batch)
        loss.backward()
"""

import logging
import functools
from typing import Callable, Any, Optional, TypeVar, cast
from contextlib import contextmanager

import torch

T = TypeVar('T')


@contextmanager
def gpu_safe_operation(operation_name: str, cleanup_on_error: bool = True):
    """
    Context manager for GPU operations with automatic cleanup on error.
    
    Catches OOM and CUDA errors, logs them with context, and optionally
    cleans up GPU memory before re-raising.
    
    Args:
        operation_name: Description of operation for error messages
        cleanup_on_error: If True, call torch.cuda.empty_cache() on GPU errors
    
    Raises:
        RuntimeError: Re-raised with additional context after cleanup
    
    Example:
        with gpu_safe_operation("Forward pass"):
            output = model(batch)
            loss = criterion(output, target)
    """
    try:
        yield
    except RuntimeError as e:
        error_msg = str(e).lower()
        
        if "out of memory" in error_msg:
            logging.error(
                f"GPU OOM during {operation_name}. "
                f"Error: {e}"
            )
            if cleanup_on_error and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU cache cleared after OOM")
            raise RuntimeError(
                f"GPU out of memory during {operation_name}. "
                f"Consider reducing batch size or model size. "
                f"Original error: {e}"
            ) from e
        
        elif "cuda" in error_msg:
            logging.error(
                f"CUDA error during {operation_name}. "
                f"Error: {e}"
            )
            if cleanup_on_error and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU cache cleared after CUDA error")
            raise RuntimeError(
                f"CUDA error during {operation_name}. "
                f"Check GPU availability and drivers. "
                f"Original error: {e}"
            ) from e
        
        else:
            # Unknown RuntimeError, re-raise as-is
            logging.error(
                f"RuntimeError during {operation_name}: {e}",
                exc_info=True
            )
            raise


@contextmanager
def model_cleanup_guard(model: Optional[torch.nn.Module] = None):
    """
    Context manager ensuring model cleanup even on error.
    
    Guarantees that GPU memory is freed even if training crashes.
    
    Args:
        model: PyTorch model to clean up (optional)
    
    Example:
        with model_cleanup_guard(model):
            train_loop(model, data_loader)
        # Model always deleted, GPU cache always cleared
    """
    try:
        yield
    finally:
        # Always cleanup, even on error
        if model is not None:
            try:
                del model
            except Exception as e:
                logging.debug(f"Failed to delete model: {e}")
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logging.debug("GPU cache cleared in cleanup guard")
            except Exception as e:
                logging.debug(f"Failed to clear GPU cache: {e}")


def log_and_reraise(
    operation: str,
    context: Optional[dict] = None,
    error_class: type = RuntimeError
) -> Callable:
    """
    Decorator that logs exceptions with context before re-raising.
    
    Args:
        operation: Description of the operation
        context: Optional dict of context variables to log
        error_class: Exception class to wrap original exception in
    
    Example:
        @log_and_reraise("Model training", context={"epoch": 5, "batch": 100})
        def train_epoch(model, loader):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context_str = ""
                if context:
                    context_str = " | ".join(f"{k}={v}" for k, v in context.items())
                    context_str = f" [{context_str}]"
                
                logging.error(
                    f"Error during {operation}{context_str}: {e}",
                    exc_info=True
                )
                
                raise error_class(
                    f"Failed during {operation}{context_str}: {e}"
                ) from e
        
        return cast(Callable[..., T], wrapper)
    return decorator


def validate_preconditions(
    model: Optional[torch.nn.Module] = None,
    data_loader: Optional[Any] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None
) -> None:
    """
    Validate common preconditions for training experiments.
    
    Raises informative errors early rather than crashing hours into training.
    
    Args:
        model: PyTorch model to validate
        data_loader: DataLoader to validate
        epochs: Number of epochs to validate
        learning_rate: Learning rate to validate
        batch_size: Batch size to validate
    
    Raises:
        ValueError: If any precondition is invalid
        TypeError: If arguments have wrong type
    
    Example:
        validate_preconditions(
            model=model,
            data_loader=train_loader,
            epochs=100,
            learning_rate=0.001
        )
    """
    if model is not None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"model must be torch.nn.Module, got {type(model).__name__}"
            )
        
        param_count = sum(p.numel() for p in model.parameters())
        if param_count == 0:
            raise ValueError("model has no parameters - cannot train")
    
    if data_loader is not None:
        try:
            loader_len = len(data_loader)
        except TypeError:
            # Some loaders don't support len()
            loader_len = None
        
        if loader_len is not None and loader_len == 0:
            raise ValueError(
                "data_loader is empty - no data to train on. "
                "Check dataset loading and filtering."
            )
    
    if epochs is not None:
        if not isinstance(epochs, int):
            raise TypeError(
                f"epochs must be int, got {type(epochs).__name__}"
            )
        if epochs <= 0:
            raise ValueError(
                f"epochs must be positive, got {epochs}"
            )
    
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate must be numeric, got {type(learning_rate).__name__}"
            )
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}"
            )
        if learning_rate > 10.0:
            logging.warning(
                f"Unusually high learning_rate: {learning_rate}. "
                f"Typical range is [1e-5, 1.0]. Double-check config."
            )
    
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be int, got {type(batch_size).__name__}"
            )
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}"
            )


def atomic_save_checkpoint(
    checkpoint: dict,
    path: str,
    operation_name: str = "checkpoint save"
) -> None:
    """
    Save PyTorch checkpoint atomically to prevent corruption.
    
    Uses temp file + atomic rename pattern to ensure checkpoint
    is either fully written or not at all (never corrupted).
    
    Args:
        checkpoint: Checkpoint dictionary to save
        path: Target file path
        operation_name: Description for error messages
    
    Raises:
        RuntimeError: If save fails
    
    Example:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        atomic_save_checkpoint(checkpoint, 'checkpoints/model_epoch_10.pt')
    """
    import tempfile
    from pathlib import Path
    
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file in same directory (ensures same filesystem)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=path_obj.parent,
            prefix='.tmp_',
            suffix='.pt'
        )
        
        try:
            # Close the file descriptor, let torch.save handle the file
            import os
            os.close(temp_fd)
            
            # Save to temp file
            torch.save(checkpoint, temp_path)
            
            # Atomic rename
            Path(temp_path).replace(path_obj)
            
            logging.debug(f"Atomically saved checkpoint: {path}")
        
        finally:
            # Clean up temp file if it still exists
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
    
    except Exception as e:
        logging.error(
            f"Failed to save checkpoint during {operation_name}: {e}",
            exc_info=True
        )
        raise RuntimeError(
            f"Checkpoint save failed during {operation_name}. "
            f"Path: {path}. Error: {e}"
        ) from e


def safe_gpu_operation(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for GPU operations with automatic error handling and cleanup.
    
    Wraps function with try/except for common GPU errors and ensures
    cache cleanup on failure.
    
    Example:
        @safe_gpu_operation
        def train_step(model, batch):
            output = model(batch)
            loss.backward()
            return loss.item()
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            if "out of memory" in error_msg:
                logging.error(f"OOM in {func.__name__}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"GPU out of memory in {func.__name__}. "
                    f"Try reducing batch size."
                ) from e
            
            elif "cuda" in error_msg:
                logging.error(f"CUDA error in {func.__name__}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA error in {func.__name__}. "
                    f"Check GPU availability."
                ) from e
            
            else:
                logging.error(
                    f"Error in {func.__name__}: {e}",
                    exc_info=True
                )
                raise
    
    return cast(Callable[..., T], wrapper)


class ErrorContext:
    """
    Context manager for adding context to errors.
    
    Example:
        with ErrorContext("Training epoch 5"):
            train_step()
        # Errors will include "During: Training epoch 5" in message
    """
    
    def __init__(self, context: str, log_entry: bool = True):
        """
        Initialize error context.
        
        Args:
            context: Description of current operation
            log_entry: If True, log when entering context
        """
        self.context = context
        self.log_entry = log_entry
    
    def __enter__(self):
        if self.log_entry:
            logging.debug(f"Entering: {self.context}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logging.error(
                f"Error during: {self.context}. "
                f"Exception: {exc_type.__name__}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            # Let exception propagate
        return False


# Convenience alias
error_context = ErrorContext
