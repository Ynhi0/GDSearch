"""
Optimizer Adapter for Optuna Tuning Compatibility

This module provides adapters to ensure hyperparameters tuned with PyTorch-native
optimizers (torch.optim.Adam, torch.optim.SGD) are compatible with custom optimizer
wrappers (AdamWrapper, SGDWrapper) used in the main experiment runner.

CRITICAL: This ensures scientific integrity by validating that tuned hyperparameters
transfer correctly between tuning and evaluation phases.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

try:
    from src.core.pytorch_optimizers import (
        SGDWrapper,
        SGDMomentumWrapper,
        AdamWrapper,
        AdamWWrapper,
        RMSPropWrapper,
    )
except ImportError:
    # Fallback for different import contexts
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.core.pytorch_optimizers import (
        SGDWrapper,
        SGDMomentumWrapper,
        AdamWrapper,
        AdamWWrapper,
        RMSPropWrapper,
    )


def build_optimizer_for_tuning(
    optimizer_name: str,
    model: nn.Module,
    params: Dict[str, Any],
    use_custom_wrappers: bool = False
) -> torch.optim.Optimizer:
    """
    Build optimizer instance for Optuna tuning.
    
    Args:
        optimizer_name: Name of optimizer ('adam', 'sgd', 'sgdmomentum', etc.)
        model: PyTorch model
        params: Dictionary of hyperparameters (lr, beta1, beta2, momentum, etc.)
        use_custom_wrappers: If True, use custom wrappers; if False, use native PyTorch
        
    Returns:
        Optimizer instance
        
    Note:
        When use_custom_wrappers=False, this builds native PyTorch optimizers for tuning.
        When use_custom_wrappers=True, this builds custom wrappers for experiments.
        Both should produce equivalent behavior for the same hyperparameters.
    """
    lr = params['lr']
    name_lower = optimizer_name.lower()
    
    if use_custom_wrappers:
        # Use custom wrappers (for experiment runner)
        if name_lower == 'adam':
            return AdamWrapper(
                model.parameters(),
                lr=lr,
                beta1=params.get('beta1', 0.9),
                beta2=params.get('beta2', 0.999),
                epsilon=params.get('epsilon', 1e-8)
            )
        elif name_lower == 'adamw':
            return AdamWWrapper(
                model.parameters(),
                lr=lr,
                weight_decay=params.get('weight_decay', 0.01)
            )
        elif name_lower in ['sgd', 'sgdmomentum']:
            if 'momentum' in params and params['momentum'] > 0:
                return SGDMomentumWrapper(
                    model.parameters(),
                    lr=lr,
                    momentum=params['momentum']
                )
            else:
                return SGDWrapper(
                    model.parameters(),
                    lr=lr
                )
        elif name_lower == 'rmsprop':
            return RMSPropWrapper(
                model.parameters(),
                lr=lr,
                alpha=params.get('alpha', 0.99),
                epsilon=params.get('epsilon', 1e-8)
            )
        else:
            raise ValueError(f"Unknown optimizer for custom wrappers: {optimizer_name}")
    
    else:
        # Use native PyTorch optimizers (for tuning)
        if name_lower == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(params.get('beta1', 0.9), params.get('beta2', 0.999)),
                eps=params.get('epsilon', 1e-8)
            )
        elif name_lower == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(params.get('beta1', 0.9), params.get('beta2', 0.999)),
                eps=params.get('epsilon', 1e-8),
                weight_decay=params.get('weight_decay', 0.01)
            )
        elif name_lower in ['sgd', 'sgdmomentum']:
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=params.get('momentum', 0.0)
            )
        elif name_lower == 'rmsprop':
            return torch.optim.RMSprop(
                model.parameters(),
                lr=lr,
                alpha=params.get('alpha', 0.99),
                eps=params.get('epsilon', 1e-8)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


def validate_optimizer_equivalence(
    optimizer_name: str,
    params: Dict[str, Any],
    test_input_size: int = 784,
    test_hidden_size: int = 256,
    test_output_size: int = 10,
    num_steps: int = 10,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that native and custom optimizer implementations produce equivalent results.
    
    This is a critical test to ensure that hyperparameters tuned with native PyTorch
    optimizers will transfer correctly to custom wrapper implementations.
    
    Args:
        optimizer_name: Name of optimizer to test
        params: Hyperparameters to test
        test_input_size: Size of test input
        test_hidden_size: Size of hidden layer
        test_output_size: Size of output
        num_steps: Number of optimization steps to test
        tolerance: Maximum allowed difference in parameters
        
    Returns:
        True if optimizers are equivalent within tolerance, False otherwise
    """
    # Create two identical models
    torch.manual_seed(42)
    model_native = nn.Sequential(
        nn.Linear(test_input_size, test_hidden_size),
        nn.ReLU(),
        nn.Linear(test_hidden_size, test_output_size)
    )
    
    torch.manual_seed(42)
    model_custom = nn.Sequential(
        nn.Linear(test_input_size, test_hidden_size),
        nn.ReLU(),
        nn.Linear(test_hidden_size, test_output_size)
    )
    
    # Build optimizers
    opt_native = build_optimizer_for_tuning(optimizer_name, model_native, params, use_custom_wrappers=False)
    opt_custom = build_optimizer_for_tuning(optimizer_name, model_custom, params, use_custom_wrappers=True)
    
    # Create test data
    torch.manual_seed(123)
    test_input = torch.randn(32, test_input_size)
    test_target = torch.randint(0, test_output_size, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Run optimization for num_steps
    for step in range(num_steps):
        # Native optimizer step
        opt_native.zero_grad()
        output_native = model_native(test_input)
        loss_native = criterion(output_native, test_target)
        loss_native.backward()
        opt_native.step()
        
        # Custom optimizer step
        opt_custom.zero_grad()
        output_custom = model_custom(test_input)
        loss_custom = criterion(output_custom, test_target)
        loss_custom.backward()
        opt_custom.step()
    
    # Compare final parameters
    max_diff = 0.0
    for p_native, p_custom in zip(model_native.parameters(), model_custom.parameters()):
        diff = torch.abs(p_native - p_custom).max().item()
        max_diff = max(max_diff, diff)
    
    equivalent = max_diff < tolerance
    
    if not equivalent:
        logging.warning(
            f"Optimizer equivalence check FAILED for {optimizer_name}: "
            f"max_diff={max_diff:.2e} > tolerance={tolerance:.2e}. "
            f"Hyperparameters tuned with native optimizer may not transfer correctly."
        )
    else:
        logging.info(
            f"Optimizer equivalence check PASSED for {optimizer_name}: "
            f"max_diff={max_diff:.2e} <= tolerance={tolerance:.2e}"
        )
    
    return equivalent


if __name__ == '__main__':
    """Test optimizer equivalence for common configurations."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("OPTIMIZER EQUIVALENCE VALIDATION")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        ('adam', {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}),
        ('adam', {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}),
        ('sgdmomentum', {'lr': 0.01, 'momentum': 0.9}),
        ('sgdmomentum', {'lr': 0.1, 'momentum': 0.9}),
    ]
    
    all_passed = True
    for opt_name, params in test_configs:
        print(f"\nTesting {opt_name} with params: {params}")
        passed = validate_optimizer_equivalence(opt_name, params, num_steps=10, tolerance=1e-6)
        if not passed:
            all_passed = False
            print(f"  ✗ FAILED")
        else:
            print(f"  ✓ PASSED")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED: Optimizers are equivalent")
        print("Hyperparameters can be safely transferred between tuning and experiments")
    else:
        print("SOME TESTS FAILED: Review optimizer implementations")
        print("WARNING: Tuned hyperparameters may not transfer correctly!")
    print("=" * 80)
