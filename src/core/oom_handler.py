"""
OOM-safe training utilities for PyTorch models.

This module provides robust out-of-memory (OOM) handling with automatic batch size
reduction, taint tracking for scientific validity, and proper error recovery.

Scientific Validity Note:
When OOM occurs and batch size is reduced, the training run is marked as "tainted"
to indicate that the run used variable batch sizes, which invalidates fair optimizer
comparisons in benchmarking studies.
"""
import logging
from typing import Tuple, Any, Callable, Optional
import torch
import torch.nn as nn


def oom_safe_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    opt_name: str = "",
    max_retries: int = 3,
    min_batch_size: int = 1
) -> Tuple[float, int, Any, bool]:
    """
    OOM-safe training step with automatic batch size reduction.
    
    When CUDA OOM occurs, this function automatically reduces the batch size
    and retries the training step. The run is marked as "tainted" to indicate
    that it used variable batch sizes, which is scientifically invalid for
    fair optimizer comparisons.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        criterion: Loss function
        inputs: Input tensor batch
        targets: Target tensor batch
        device: torch.device
        opt_name: Optimizer name (for SAM handling)
        max_retries: Maximum OOM recovery attempts
        min_batch_size: Minimum batch size before giving up
        
    Returns:
        Tuple of (loss_value, actual_batch_size, outputs, tainted)
        - loss_value: Scalar loss value
        - actual_batch_size: Final batch size used
        - outputs: Model outputs
        - tainted: True if OOM recovery reduced batch size (run is invalid)
        
    Raises:
        RuntimeError: If OOM recovery fails after max_retries or batch too small
    """
    current_inputs = inputs
    current_targets = targets
    retries = 0
    original_batch_size = inputs.size(0)
    tainted = False
    
    while retries < max_retries:
        try:
            current_inputs = current_inputs.to(device)
            current_targets = current_targets.to(device)
            
            # Handle SAM optimizer (requires closure)
            if 'SAM' in opt_name.upper():
                def closure():
                    optimizer.zero_grad()
                    outputs = model(current_inputs)
                    loss = criterion(outputs, current_targets)
                    loss.backward()
                    return loss
                
                # CRITICAL FIX: SAM requires the actual closure, not a dummy lambda
                # SAM will call closure() internally to compute adversarial gradients
                loss = optimizer.step(closure)
                outputs = model(current_inputs)
                loss_value = float(loss.item()) if hasattr(loss, 'item') else float(loss)
                return loss_value, current_inputs.size(0), outputs, tainted
            
            else:
                # Standard optimizer step
                optimizer.zero_grad()
                outputs = model(current_inputs)
                loss = criterion(outputs, current_targets)
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Check for loss divergence
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("Loss divergence detected: %f", loss.item())
                    return float('inf'), current_inputs.size(0), outputs, tainted
                
                return loss.item(), current_inputs.size(0), outputs, tainted
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                retries += 1
                torch.cuda.empty_cache()
                
                old_size = current_inputs.size(0)
                new_size = max(min_batch_size, old_size // 2)
                
                # Check BatchNorm compatibility BEFORE reduction
                if new_size < 2:
                    logging.error(
                        "Cannot reduce batch to %d (BatchNorm requires >= 2)",
                        new_size
                    )
                    raise RuntimeError(
                        "Batch size too small for BatchNorm layers"
                    ) from e
                
                if new_size < min_batch_size:
                    logging.error(
                        "OOM: Cannot reduce batch below %d",
                        min_batch_size
                    )
                    raise
                
                # Mark run as tainted and log warning
                tainted = True
                logging.warning(
                    "SCIENTIFIC INTEGRITY WARNING: Run Tainted - "
                    "Batch size reduced from %d to %d due to OOM",
                    original_batch_size,
                    new_size
                )
                logging.warning(
                    "    CUDA OOM! Reducing batch: %d->%d (retry %d/%d)",
                    old_size,
                    new_size,
                    retries,
                    max_retries
                )
                logging.warning(
                    "    This run uses variable batch size and should be "
                    "excluded from fair comparisons."
                )
                
                # Slice the batch
                current_inputs = inputs[:new_size]
                current_targets = targets[:new_size]
                
                # Clear optimizer gradients
                optimizer.zero_grad(set_to_none=True)
            else:
                raise
    
    logging.error("OOM recovery failed after %d retries", max_retries)
    raise RuntimeError(f"CUDA OOM after {max_retries} recovery attempts")


def clear_gpu_memory(force: bool = False):
    """
    Clear GPU memory between experiments to prevent fragmentation and OOM.
    
    This is critical for long-running benchmark suites to:
    - Prevent cumulative memory leaks
    - Avoid fragmentation
    - Ensure consistent performance
    - Prevent OOM crashes
    
    Args:
        force: If True, perform aggressive cleanup
    """
    if torch.cuda.is_available():
        # Synchronize all CUDA streams
        torch.cuda.synchronize()
        
        # Empty the cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if force:
            # Aggressive cleanup: clear all caches
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        
        # Log memory state
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            free = total - allocated
            logging.info(
                "GPU memory cleaned: %.1fMB used, %.1fMB free",
                allocated,
                free
            )
        except Exception as e:
            logging.debug("Could not log GPU memory stats: %s", e)
