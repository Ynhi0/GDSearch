"""
Loss and gradient sanity checks for training loops.

Validates loss values and gradients to catch NaN/inf early.
"""

import logging
from typing import Optional
import torch


def validate_loss(
    loss: torch.Tensor,
    step: Optional[int] = None,
    context: str = "training",
    raise_on_invalid: bool = True
) -> bool:
    """
    Validate that loss is finite (not NaN or inf).
    
    Should be called after loss computation, before loss.backward().
    
    Args:
        loss: Loss tensor to validate
        step: Optional step number for error messages
        context: Description of where loss occurred
        raise_on_invalid: If True, raise ValueError on invalid loss
    
    Returns:
        True if loss is valid, False otherwise
    
    Raises:
        ValueError: If loss is invalid and raise_on_invalid=True
    
    Example:
        >>> loss = criterion(output, target)
        >>> validate_loss(loss, step=batch_idx, context="MNIST training")
        >>> loss.backward()
    """
    if not torch.isfinite(loss):
        step_str = f" at step {step}" if step is not None else ""
        error_msg = (
            f"Invalid loss detected during {context}{step_str}: "
            f"loss={loss.item():.6f}"
        )
        
        if raise_on_invalid:
            logging.error(error_msg)
            raise ValueError(error_msg)
        else:
            logging.warning(error_msg)
            return False
    
    return True


def validate_gradients(
    model: torch.nn.Module,
    step: Optional[int] = None,
    context: str = "training",
    raise_on_invalid: bool = True
) -> bool:
    """
    Validate that all model gradients are finite.
    
    Should be called after loss.backward(), before optimizer.step().
    
    Args:
        model: Model to check gradients
        step: Optional step number for error messages
        context: Description of where gradients occurred
        raise_on_invalid: If True, raise ValueError on invalid gradients
    
    Returns:
        True if all gradients are valid, False otherwise
    
    Raises:
        ValueError: If gradients are invalid and raise_on_invalid=True
    
    Example:
        >>> loss.backward()
        >>> validate_gradients(model, step=batch_idx, context="MNIST training")
        >>> optimizer.step()
    """
    invalid_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            invalid_params.append(name)
    
    if invalid_params:
        step_str = f" at step {step}" if step is not None else ""
        error_msg = (
            f"Invalid gradients detected during {context}{step_str}. "
            f"Affected parameters: {', '.join(invalid_params[:5])}"
            + (f" and {len(invalid_params)-5} more" if len(invalid_params) > 5 else "")
        )
        
        if raise_on_invalid:
            logging.error(error_msg)
            raise ValueError(error_msg)
        else:
            logging.warning(error_msg)
            return False
    
    return True


__all__ = ['validate_loss', 'validate_gradients']
