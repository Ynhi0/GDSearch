"""
Gradient health monitoring utilities for Kaggle benchmarks.
"""

import torch
import logging


def check_gradient_health(model, epoch=None, threshold=1e3, context=""):
    """
    Quick gradient health check for training loops.

    Args:
        model: PyTorch model
        epoch: Current epoch number (optional, for logging)
        threshold: Gradient norm explosion threshold
        context: Context string for logging (e.g., "CIFAR-10", "NLP")

    Returns:
        grad_norm: Total gradient norm, or inf if bad gradients detected
    """
    try:
        grad_norm = 0.0
        has_bad_grad = False

        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    epoch_str = f" at epoch {epoch}" if epoch is not None else ""
                    logging.warning(f"NaN/Inf gradient detected{epoch_str} ({context})")
                    has_bad_grad = True
                    break
                grad_norm += param.grad.data.norm(2).item() ** 2

        if not has_bad_grad:
            grad_norm = grad_norm ** 0.5
            if grad_norm > threshold:
                epoch_str = f" at epoch {epoch}" if epoch is not None else ""
                logging.warning(f"Large gradient norm{epoch_str}: {grad_norm:.2e} ({context})")

        return grad_norm if not has_bad_grad else float('inf')
    except Exception as e:
        logging.debug(f"Gradient check failed ({context}): {e}")
        return 0.0
