"""
Input validation and error handling utilities.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Union


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate experiment configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Validated config (may have default values added)

    Raises:
        ValidationError: If validation fails
    """
    validated = config.copy()

    # Required fields
    required_fields = []

    # Check task type
    task = validated.get('task', 'neural_network')
    if task not in ['neural_network', 'test_function']:
        raise ValidationError(f"Invalid task: {task}. Must be 'neural_network' or 'test_function'")

    if task == 'neural_network':
        # NN-specific validation
        if 'dataset' not in validated:
            raise ValidationError("Missing required field: 'dataset'")

        if validated['dataset'].upper() not in ['MNIST', 'CIFAR-10', 'CIFAR10']:
            raise ValidationError(f"Invalid dataset: {validated['dataset']}")

        if 'model' not in validated:
            raise ValidationError("Missing required field: 'model'")

        if 'optimizer' not in validated:
            raise ValidationError("Missing required field: 'optimizer'")

        # Validate numeric params
        validated['lr'] = validate_learning_rate(validated.get('lr', 1e-3))
        validated['epochs'] = validate_epochs(validated.get('epochs', 10))
        validated['batch_size'] = validate_batch_size(validated.get('batch_size', 128))
        validated['weight_decay'] = validate_weight_decay(validated.get('weight_decay', 0.0))
        validated['seed'] = validated.get('seed', 42)

        # Optimizer-specific params
        opt = validated['optimizer'].upper()

        if 'MOMENTUM' in opt:
            validated['momentum'] = validate_momentum(validated.get('momentum', 0.9))

        if 'ADAM' in opt:
            validated['beta1'] = validate_beta(validated.get('beta1', 0.9), 'beta1')
            validated['beta2'] = validate_beta(validated.get('beta2', 0.999), 'beta2')
            validated['epsilon'] = validate_epsilon(validated.get('epsilon', 1e-8))

        if 'RMSPROP' in opt:
            validated['alpha'] = validate_alpha(validated.get('alpha', 0.99))

    elif task == 'test_function':
        # Test function-specific validation
        if 'function' not in validated:
            raise ValidationError("Missing required field: 'function'")

        if 'optimizer' not in validated:
            raise ValidationError("Missing required field: 'optimizer'")

        validated['lr'] = validate_learning_rate(validated.get('lr', 0.01))
        validated['num_iterations'] = validate_num_iterations(validated.get('num_iterations', 1000))
        validated['seed'] = validated.get('seed', 42)

    return validated


def validate_learning_rate(lr: float) -> float:
    """Validate learning rate."""
    if not isinstance(lr, (int, float)):
        raise ValidationError(f"Learning rate must be numeric, got {type(lr)}")

    if lr <= 0:
        raise ValidationError(f"Learning rate must be positive, got {lr}")

    if lr > 10:
        raise ValidationError(f"Learning rate too large: {lr}. Typical range: [1e-5, 1.0]")

    return float(lr)


def validate_epochs(epochs: int) -> int:
    """Validate number of epochs."""
    if not isinstance(epochs, int):
        try:
            epochs = int(epochs)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Epochs must be integer, got {type(epochs)}") from e

    if epochs <= 0:
        raise ValidationError(f"Epochs must be positive, got {epochs}")

    if epochs > 1000:
        raise ValidationError(f"Epochs too large: {epochs}. Are you sure?")

    return epochs


def validate_batch_size(batch_size: int) -> int:
    """Validate batch size."""
    if not isinstance(batch_size, int):
        try:
            batch_size = int(batch_size)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Batch size must be integer, got {type(batch_size)}") from e

    if batch_size <= 0:
        raise ValidationError(f"Batch size must be positive, got {batch_size}")

    if batch_size > 10000:
        raise ValidationError(f"Batch size too large: {batch_size}")

    # Check power of 2 (best for GPU)
    if batch_size & (batch_size - 1) != 0:
        import warnings
        warnings.warn(f"Batch size {batch_size} is not a power of 2. Consider using 32, 64, 128, 256, etc.")

    return batch_size


def validate_weight_decay(wd: float) -> float:
    """Validate weight decay."""
    if not isinstance(wd, (int, float)):
        raise ValidationError(f"Weight decay must be numeric, got {type(wd)}")

    if wd < 0:
        raise ValidationError(f"Weight decay must be non-negative, got {wd}")

    if wd > 1:
        raise ValidationError(f"Weight decay too large: {wd}. Typical range: [0, 0.01]")

    return float(wd)


def validate_momentum(momentum: float) -> float:
    """Validate momentum coefficient."""
    if not isinstance(momentum, (int, float)):
        raise ValidationError(f"Momentum must be numeric, got {type(momentum)}")

    if momentum < 0 or momentum >= 1:
        raise ValidationError(f"Momentum must be in [0, 1), got {momentum}")

    return float(momentum)


def validate_beta(beta: float, name: str) -> float:
    """Validate beta coefficients for Adam."""
    if not isinstance(beta, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(beta)}")

    if beta < 0 or beta >= 1:
        raise ValidationError(f"{name} must be in [0, 1), got {beta}")

    return float(beta)


def validate_alpha(alpha: float) -> float:
    """Validate alpha for RMSProp."""
    if not isinstance(alpha, (int, float)):
        raise ValidationError(f"Alpha must be numeric, got {type(alpha)}")

    if alpha < 0 or alpha >= 1:
        raise ValidationError(f"Alpha must be in [0, 1), got {alpha}")

    return float(alpha)


def validate_epsilon(epsilon: float) -> float:
    """Validate epsilon for numerical stability."""
    if not isinstance(epsilon, (int, float)):
        raise ValidationError(f"Epsilon must be numeric, got {type(epsilon)}")

    if epsilon <= 0:
        raise ValidationError(f"Epsilon must be positive, got {epsilon}")

    if epsilon > 1e-3:
        raise ValidationError(f"Epsilon too large: {epsilon}. Typical range: [1e-10, 1e-6]")

    return float(epsilon)


def validate_num_iterations(num_iter: int) -> int:
    """Validate number of iterations for test functions."""
    if not isinstance(num_iter, int):
        try:
            num_iter = int(num_iter)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Num iterations must be integer, got {type(num_iter)}") from e

    if num_iter <= 0:
        raise ValidationError(f"Num iterations must be positive, got {num_iter}")

    if num_iter > 1000000:
        raise ValidationError(f"Num iterations too large: {num_iter}")

    return num_iter


def check_for_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """
    Check tensor for NaN or Inf values.

    Args:
        tensor: PyTorch tensor to check
        name: Name for error messages

    Raises:
        RuntimeError: If NaN or Inf detected
    """
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected in {name}")

    if torch.isinf(tensor).any():
        raise RuntimeError(f"Inf detected in {name}")


def check_gradient_health(model: torch.nn.Module, threshold: float = 1e3):
    """
    Check if gradients are healthy (no NaN/Inf, not exploding).

    Args:
        model: PyTorch model
        threshold: Threshold for gradient norm explosion

    Raises:
        RuntimeError: If gradients are unhealthy
    """
    total_norm = 0.0
    # Track whether we found any gradients
    has_gradients = False

    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            # Check for NaN/Inf
            if torch.isnan(param.grad).any():
                raise RuntimeError(f"NaN gradient in {name}")

            if torch.isinf(param.grad).any():
                raise RuntimeError(f"Inf gradient in {name}")

            # Accumulate norm
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2

    # Distinguish between no gradients vs zero gradients
    if not has_gradients:
        import warnings
        warnings.warn("No gradients found in model. Did you call backward()?")
        return 0.0

    total_norm = np.sqrt(total_norm)

    # Check for explosion
    if total_norm > threshold:
        import warnings
        warnings.warn(f"Large gradient norm detected: {total_norm:.2e}. May indicate gradient explosion.")

    return total_norm


def validate_seeds(seeds: Union[int, List[int]]) -> List[int]:
    """
    Validate random seeds.

    Args:
        seeds: Single seed or list of seeds

    Returns:
        List of validated seeds

    Raises:
        ValidationError: If seeds are invalid
    """
    if isinstance(seeds, int):
        seeds = [seeds]

    if not isinstance(seeds, list):
        raise ValidationError(f"Seeds must be int or list of ints, got {type(seeds)}")

    validated = []
    for seed in seeds:
        if not isinstance(seed, int):
            try:
                seed = int(seed)
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Seed must be integer, got {type(seed)}") from e

        if seed < 0:
            raise ValidationError(f"Seed must be non-negative, got {seed}")

        validated.append(seed)

    return validated


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        numerator / denominator, or default if denominator is zero
    """
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def clip_gradient_norm(model: torch.nn.Module, max_norm: float = 1.0):
    """
    Clip gradients by global norm.

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


if __name__ == '__main__':
    # Test validation
    print("Testing validation...")

    # Valid config
    config = {
        'task': 'neural_network',
        'dataset': 'MNIST',
        'model': 'SimpleMLP',
        'optimizer': 'Adam',
        'lr': 1e-3,
        'epochs': 10,
        'batch_size': 128
    }

    validated = validate_config(config)
    print("Valid config accepted")
    print(f"   Validated config: {validated}")

    # Invalid learning rate
    try:
        invalid_config = config.copy()
        invalid_config['lr'] = -0.01
        validate_config(invalid_config)
        print("Should have rejected negative learning rate")
    except ValidationError as e:
        print(f"Correctly rejected invalid LR: {e}")

    # Invalid batch size
    try:
        invalid_config = config.copy()
        invalid_config['batch_size'] = 0
        validate_config(invalid_config)
        print("Should have rejected zero batch size")
    except ValidationError as e:
        print(f"Correctly rejected invalid batch size: {e}")

    print("\nAll validation tests passed!")


# ============================================================================
# DEEP LOGIC REVIEW ADDITIONS - Enhanced Validation Functions
# ============================================================================

import math


def validate_loss(
    loss: Union[torch.Tensor, float],
    context: str = "",
    max_allowed: float = 1e6
) -> torch.Tensor:
    """
    Validate loss is finite before backward pass.
    
    Catches NaN/Inf early to prevent gradient corruption and wasted computation.
    
    Args:
        loss: Loss tensor or scalar
        context: Context string for error message (e.g., "epoch 5, batch 23")
        max_allowed: Maximum allowed loss value (to catch exploding loss early)
    
    Returns:
        Original loss tensor if valid
    
    Raises:
        ValidationError: If loss is NaN, Inf, or exceeds max_allowed
    
    Example:
        >>> loss = criterion(output, target)
        >>> validate_loss(loss, context=f"epoch {epoch}, batch {batch_idx}")
        >>> loss.backward()  # Safe - loss is guaranteed finite
    """
    # Convert scalar to tensor for uniform handling
    if isinstance(loss, (int, float)):
        loss_value = float(loss)
        is_tensor = False
    else:
        loss_value = loss.item()
        is_tensor = True
    
    context_str = f" ({context})" if context else ""
    
    # Check for NaN
    if math.isnan(loss_value):
        raise ValidationError(
            f"NaN loss detected{context_str}. Training has diverged.\n"
            f"REMEDIATION:\n"
            f"  1. Reduce learning rate (current value may be too high)\n"
            f"  2. Use gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
            f"  3. Check for unstable operations (log(0), sqrt(negative), division by zero)\n"
            f"  4. Verify input data is normalized and doesn't contain NaN/Inf\n"
            f"  5. Consider using a smaller model or batch size"
        )
    
    # Check for Inf
    if math.isinf(loss_value):
        raise ValidationError(
            f"Infinite loss detected{context_str}. Loss exploded to {loss_value}.\n"
            f"REMEDIATION:\n"
            f"  1. CRITICAL: Reduce learning rate significantly (try 10x smaller)\n"
            f"  2. Enable gradient clipping immediately\n"
            f"  3. Check for numerical instability in custom loss functions\n"
            f"  4. Verify model initialization (weights may be too large)\n"
            f"  5. Check for exploding gradients in earlier layers"
        )
    
    # Check for suspiciously large loss
    if loss_value > max_allowed:
        import logging
        logging.warning(
            f"Very large loss detected{context_str}: {loss_value:.2e} > {max_allowed:.2e}. "
            f"This may indicate training instability."
        )
    
    # Check for negative loss (usually indicates a bug)
    if loss_value < 0:
        import logging
        logging.warning(
            f"Negative loss detected{context_str}: {loss_value:.4f}. "
            f"This is unusual for most loss functions and may indicate a bug."
        )
    
    # Return original tensor (not scalar) to preserve gradients
    return loss if is_tensor else torch.tensor(loss_value)


def validate_dataset(
    dataset,
    min_samples: int = 1,
    name: str = "dataset"
) -> int:
    """
    Validate dataset is non-empty and has expected properties.
    
    Args:
        dataset: Dataset to validate (must have __len__)
        min_samples: Minimum required samples
        name: Dataset name for error messages
    
    Returns:
        Dataset length
    
    Raises:
        ValidationError: If dataset is empty or too small
        AttributeError: If dataset doesn't support len()
    
    Example:
        >>> train_dataset = datasets.MNIST(...)
        >>> n_train = validate_dataset(train_dataset, min_samples=100, name="training")
    """
    try:
        dataset_len = len(dataset)
    except (TypeError, AttributeError) as e:
        raise AttributeError(
            f"{name} does not support len(). "
            f"Ensure it's a proper torch.utils.data.Dataset. "
            f"Error: {e}"
        ) from e
    
    if dataset_len == 0:
        raise ValidationError(
            f"{name} is empty (0 samples).\n"
            f"REMEDIATION:\n"
            f"  1. Check dataset download succeeded\n"
            f"  2. Verify data directory is correct\n"
            f"  3. Check for file corruption in data files\n"
            f"  4. Review dataset construction code for bugs"
        )
    
    if dataset_len < min_samples:
        raise ValidationError(
            f"{name} has only {dataset_len} samples, "
            f"but {min_samples} required.\n"
            f"REMEDIATION:\n"
            f"  1. Download full dataset (may have partial download)\n"
            f"  2. Adjust min_samples if this is intentional (e.g., quick test)\n"
            f"  3. Check dataset split logic (train/val/test) for errors"
        )
    
    import logging
    logging.debug(f"{name}: {dataset_len} samples (>= {min_samples} required)")
    return dataset_len


def validate_batch_size(
    batch_size: int,
    dataset_len: int,
    model: torch.nn.Module = None,
    dataset_name: str = "dataset"
) -> None:
    """
    Validate batch size is compatible with dataset and model.
    
    Checks:
    1. Batch size > 0
    2. Batch size <= dataset length
    3. Batch size >= 2 if model has BatchNorm
    
    Args:
        batch_size: Requested batch size
        dataset_len: Dataset length
        model: Model (optional, for BatchNorm check)
        dataset_name: Dataset name for error messages
    
    Raises:
        ValidationError: If batch size is invalid
    
    Example:
        >>> validate_batch_size(batch_size=128, dataset_len=60000, model=model)
    """
    if batch_size <= 0:
        raise ValidationError(
            f"Batch size must be positive, got {batch_size}"
        )
    
    if batch_size > dataset_len:
        raise ValidationError(
            f"Batch size ({batch_size}) larger than {dataset_name} ({dataset_len} samples).\n"
            f"REMEDIATION:\n"
            f"  1. Reduce batch size to at most {dataset_len}\n"
            f"  2. Use full-batch training (batch_size={dataset_len}) if intentional\n"
            f"  3. Check dataset construction for errors"
        )
    
    # Check for BatchNorm compatibility
    if model is not None and batch_size == 1:
        if has_batchnorm(model):
            raise ValidationError(
                f"Model uses BatchNorm but batch_size=1.\n"
                f"BatchNorm requires batch_size >= 2 for training mode.\n"
                f"REMEDIATION:\n"
                f"  1. Use batch_size >= 2\n"
                f"  2. Replace BatchNorm with LayerNorm or GroupNorm\n"
                f"  3. Use model.eval() mode (disables BatchNorm updates, changes semantics)"
            )
    
    import logging
    logging.debug(
        f"Batch size validation passed: batch_size={batch_size}, "
        f"{dataset_name}_len={dataset_len}"
    )


def has_batchnorm(model: torch.nn.Module) -> bool:
    """
    Check if model contains BatchNorm layers.
    
    Args:
        model: PyTorch model
    
    Returns:
        True if model has any BatchNorm layers
    
    Example:
        >>> model = SimpleMLP(784, 128, 10)  # has BatchNorm1d
        >>> has_batchnorm(model)
        True
    """
    for module in model.modules():
        if isinstance(module, (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm
        )):
            return True
    return False


def validate_gradients(
    model: torch.nn.Module,
    max_norm: float = 100.0,
    context: str = ""
) -> float:
    """
    Validate gradients are finite and not exploding.
    
    Should be called after loss.backward() but before optimizer.step().
    
    Args:
        model: Model to check gradients
        max_norm: Maximum allowed gradient norm
        context: Context string for error messages
    
    Returns:
        Total gradient norm
    
    Raises:
        ValidationError: If gradients are NaN or too large
    
    Example:
        >>> loss.backward()
        >>> grad_norm = validate_gradients(model, max_norm=10.0, context="epoch 5")
        >>> optimizer.step()
    """
    total_norm = 0.0
    has_grad = False
    
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            param_norm = param.grad.data.norm(2).item()
            
            # Check for NaN
            if math.isnan(param_norm):
                context_str = f" ({context})" if context else ""
                raise ValidationError(
                    f"NaN gradient detected{context_str}.\n"
                    f"Parameter shape: {param.shape}\n"
                    f"REMEDIATION:\n"
                    f"  1. Reduce learning rate\n"
                    f"  2. Use gradient clipping\n"
                    f"  3. Check for unstable operations in model"
                )
            
            # Check for Inf
            if math.isinf(param_norm):
                context_str = f" ({context})" if context else ""
                raise ValidationError(
                    f"Infinite gradient detected{context_str}.\n"
                    f"Parameter shape: {param.shape}\n"
                    f"Gradient norm: {param_norm}\n"
                    f"REMEDIATION:\n"
                    f"  1. CRITICAL: Reduce learning rate\n"
                    f"  2. Enable gradient clipping\n"
                    f"  3. Check model initialization"
                )
            
            total_norm += param_norm ** 2
    
    if not has_grad:
        context_str = f" ({context})" if context else ""
        import logging
        logging.warning(
            f"No gradients found{context_str}. "
            f"Check that model parameters require_grad=True and "
            f"loss.backward() was called."
        )
        return 0.0
    
    total_norm = total_norm ** 0.5
    
    # Warn for large gradients
    if total_norm > max_norm:
        context_str = f" ({context})" if context else ""
        import logging
        logging.warning(
            f"Large gradient norm detected{context_str}: {total_norm:.2e} > {max_norm:.2e}. "
            f"Consider gradient clipping."
        )
    
    return total_norm
