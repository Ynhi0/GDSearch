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
    # CRITICAL FIX: Prevent SAM state corruption
    # SAM moves weights during the step. If OOM occurs mid-step, weights are corrupted.
    # We must disable retry for SAM - better to crash than train on corrupted weights.
    if "SAM" in optimizer.__class__.__name__:
        logging.warning(
            "CRITICAL: SAM optimizer detected. OOM retry disabled to prevent state corruption. "
            "If OOM occurs, the run will fail immediately. Reduce batch size manually."
        )
        # Bypass retry logic - execute SAM step directly
        inputs_device = inputs.to(device)
        targets_device = targets.to(device)
        
        def closure():
            optimizer.zero_grad()
            outputs = model(inputs_device)
            loss = criterion(outputs, targets_device)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        
        if loss is None:
            raise RuntimeError(
                f"SAM optimizer step returned None. Expected loss tensor. "
                f"Check SAM implementation: {type(optimizer).__name__}"
            )
        
        with torch.no_grad():
            outputs = model(inputs_device)
        
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.item())
        elif isinstance(loss, (int, float)):
            loss_value = float(loss)
        else:
            raise TypeError(
                f"SAM step returned unexpected type: {type(loss)}. "
                f"Expected torch.Tensor or numeric scalar."
            )
        
        return loss_value, inputs.size(0), outputs, False
    
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
                
                # Validate closure return type
                if loss is None:
                    raise RuntimeError(
                        f"SAM optimizer step returned None. Expected loss tensor. "
                        f"Check SAM implementation: {type(optimizer).__name__}"
                    )
                
                # Get outputs after SAM step (parameters have been updated)
                with torch.no_grad():
                    outputs = model(current_inputs)
                
                # Extract scalar loss value with proper type handling
                if isinstance(loss, torch.Tensor):
                    loss_value = float(loss.item())
                elif isinstance(loss, (int, float)):
                    loss_value = float(loss)
                else:
                    raise TypeError(
                        f"SAM step returned unexpected type: {type(loss)}. "
                        f"Expected torch.Tensor or numeric scalar."
                    )
                
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
                # CRITICAL FIX: Provide more graceful handling for BatchNorm constraints
                if new_size < 2:
                    # Try to switch model to eval mode as last resort
                    # BatchNorm in eval mode uses running stats and doesn't require batch size >= 2
                    try:
                        was_training = model.training
                        model.eval()
                        logging.warning(
                            "CRITICAL: Batch size %d too small for BatchNorm in training mode. "
                            "Temporarily switching to eval mode to avoid crash. "
                            "This run is TAINTED and should be excluded from analysis.",
                            new_size
                        )
                        # Set tainted flag since we're mixing train/eval modes
                        tainted = True
                        
                        # Process the batch in eval mode, then restore
                        current_inputs_small = inputs[:new_size]
                        current_targets_small = targets[:new_size]
                        
                        optimizer.zero_grad(set_to_none=True)
                        current_inputs_small = current_inputs_small.to(device)
                        current_targets_small = current_targets_small.to(device)
                        
                        outputs = model(current_inputs_small)
                        loss = criterion(outputs, current_targets_small)
                        loss.backward()
                        optimizer.step()
                        
                        # Restore training mode before returning
                        if was_training:
                            model.train()
                        
                        # Check for loss divergence
                        if torch.isnan(loss) or torch.isinf(loss):
                            logging.warning("Loss divergence detected: %f", loss.item())
                            return float('inf'), new_size, outputs, tainted
                        
                        return loss.item(), new_size, outputs, tainted
                        
                    except Exception as eval_error:
                        # Restore training mode before raising
                        if was_training:
                            model.train()
                        logging.error(
                            "Cannot reduce batch to %d (BatchNorm requires >= 2) "
                            "and failed to process in eval mode: %s",
                            new_size, eval_error
                        )
                        raise RuntimeError(
                            "Batch size too small for BatchNorm layers and eval mode fallback failed"
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
