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
        except:
            raise ValidationError(f"Epochs must be integer, got {type(epochs)}")
    
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
        except:
            raise ValidationError(f"Batch size must be integer, got {type(batch_size)}")
    
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
        except:
            raise ValidationError(f"Num iterations must be integer, got {type(num_iter)}")
    
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
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check for NaN/Inf
            if torch.isnan(param.grad).any():
                raise RuntimeError(f"NaN gradient in {name}")
            
            if torch.isinf(param.grad).any():
                raise RuntimeError(f"Inf gradient in {name}")
            
            # Accumulate norm
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    
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
            except:
                raise ValidationError(f"Seed must be integer, got {type(seed)}")
        
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
    print("✅ Valid config accepted")
    print(f"   Validated config: {validated}")
    
    # Invalid learning rate
    try:
        invalid_config = config.copy()
        invalid_config['lr'] = -0.01
        validate_config(invalid_config)
        print("❌ Should have rejected negative learning rate")
    except ValidationError as e:
        print(f"✅ Correctly rejected invalid LR: {e}")
    
    # Invalid batch size
    try:
        invalid_config = config.copy()
        invalid_config['batch_size'] = 0
        validate_config(invalid_config)
        print("❌ Should have rejected zero batch size")
    except ValidationError as e:
        print(f"✅ Correctly rejected invalid batch size: {e}")
    
    print("\n✅ All validation tests passed!")
