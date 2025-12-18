"""
Out-of-Memory (OOM) handling utilities for GPU training.

Provides robust OOM recovery with automatic batch size reduction and taint tracking
to ensure experiment validity.
"""
import logging
from typing import Tuple, Any
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def oom_safe_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    opt_name: str = "",
    max_retries: int = 3,
    min_batch_size: int = 1
) -> Tuple[float, int, Any, bool]:
    """
    Execute a single training step with OOM recovery.
    
    If CUDA OOM occurs, automatically reduces batch size and retries.
    Returns a tainted flag to indicate if OOM recovery was used.
    
    Args:
        model: Neural network model
        optimizer: Optimizer instance
        criterion: Loss function
        inputs: Input batch tensor
        targets: Target batch tensor
        device: Device to run on (cuda/cpu)
        opt_name: Optimizer name for logging
        max_retries: Maximum number of OOM recovery attempts
        min_batch_size: Minimum batch size before giving up
        
    Returns:
        Tuple of (loss_value, effective_batch_size, outputs, tainted)
        - loss_value: Scalar loss value
        - effective_batch_size: Actual batch size used (may be reduced)
        - outputs: Model outputs from forward pass
        - tainted: True if OOM recovery was triggered
    """
    original_batch_size = inputs.size(0)
    current_batch_size = original_batch_size
    retry_count = 0
    tainted = False
    
    while retry_count < max_retries:
        try:
            # Move data to device
            inputs_device = inputs[:current_batch_size].to(device)
            targets_device = targets[:current_batch_size].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs_device)
            loss = criterion(outputs, targets_device)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Success - return results
            loss_value = loss.item()
            
            if tainted:
                logger.warning(
                    f"{opt_name}: OOM recovered - reduced batch from "
                    f"{original_batch_size} to {current_batch_size}"
                )
            
            return loss_value, current_batch_size, outputs, tainted
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Reduce batch size
                new_batch_size = max(current_batch_size // 2, min_batch_size)
                
                if new_batch_size < min_batch_size or new_batch_size == current_batch_size:
                    logger.error(
                        f"{opt_name}: OOM recovery failed - batch size {current_batch_size} "
                        f"cannot be reduced further (min={min_batch_size})"
                    )
                    raise RuntimeError(
                        f"CUDA OOM with batch_size={current_batch_size}, cannot reduce further"
                    ) from e
                
                logger.warning(
                    f"{opt_name}: CUDA OOM detected - reducing batch from "
                    f"{current_batch_size} to {new_batch_size} (retry {retry_count + 1}/{max_retries})"
                )
                
                current_batch_size = new_batch_size
                retry_count += 1
                tainted = True
                
            else:
                # Not an OOM error - re-raise
                raise
    
    # Max retries exceeded
    raise RuntimeError(
        f"{opt_name}: OOM recovery failed after {max_retries} retries "
        f"(final batch_size={current_batch_size})"
    )


def oom_safe_eval_step(
    model: nn.Module,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    max_retries: int = 3,
    min_batch_size: int = 1
) -> Tuple[float, Any, int, bool]:
    """
    Execute a single evaluation step with OOM recovery.
    
    Similar to oom_safe_train_step but for evaluation (no gradient computation).
    
    Args:
        model: Neural network model
        criterion: Loss function
        inputs: Input batch tensor
        targets: Target batch tensor
        device: Device to run on (cuda/cpu)
        max_retries: Maximum number of OOM recovery attempts
        min_batch_size: Minimum batch size before giving up
        
    Returns:
        Tuple of (loss_value, outputs, effective_batch_size, tainted)
    """
    original_batch_size = inputs.size(0)
    current_batch_size = original_batch_size
    retry_count = 0
    tainted = False
    
    model.eval()
    
    while retry_count < max_retries:
        try:
            with torch.no_grad():
                inputs_device = inputs[:current_batch_size].to(device)
                targets_device = targets[:current_batch_size].to(device)
                
                outputs = model(inputs_device)
                loss = criterion(outputs, targets_device)
                
                if tainted:
                    logger.warning(
                        f"Eval OOM recovered - reduced batch from "
                        f"{original_batch_size} to {current_batch_size}"
                    )
                
                return loss.item(), outputs, current_batch_size, tainted
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                new_batch_size = max(current_batch_size // 2, min_batch_size)
                
                if new_batch_size < min_batch_size or new_batch_size == current_batch_size:
                    logger.error(
                        f"Eval OOM recovery failed - batch size {current_batch_size} "
                        f"cannot be reduced further"
                    )
                    raise RuntimeError(
                        f"CUDA OOM in eval with batch_size={current_batch_size}"
                    ) from e
                
                logger.warning(
                    f"Eval CUDA OOM - reducing batch from {current_batch_size} "
                    f"to {new_batch_size} (retry {retry_count + 1}/{max_retries})"
                )
                
                current_batch_size = new_batch_size
                retry_count += 1
                tainted = True
            else:
                raise
    
    raise RuntimeError(
        f"Eval OOM recovery failed after {max_retries} retries "
        f"(final batch_size={current_batch_size})"
    )
