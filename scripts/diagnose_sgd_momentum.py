"""
Diagnostic script to identify the exact difference between 
PyTorch SGD+Momentum and custom SGDMomentum implementation.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.optimizers import SGDMomentum as CustomSGDMomentum
from src.core.pytorch_optimizers import SGDMomentumWrapper

# Create identical models
torch.manual_seed(42)
model_pytorch = nn.Linear(10, 1, bias=False)
torch.manual_seed(42)
model_custom = nn.Linear(10, 1, bias=False)

# Verify weights are identical
print("Initial weight difference:", 
      torch.abs(model_pytorch.weight - model_custom.weight).max().item())

# Create optimizers
pytorch_opt = torch.optim.SGD(model_pytorch.parameters(), lr=0.01, momentum=0.5)
custom_opt = SGDMomentumWrapper(model_custom.parameters(), lr=0.01, momentum=0.5)

# Create test data
torch.manual_seed(123)
x = torch.randn(5, 10)
y = torch.randn(5, 1)

# Do 3 steps and track states
for step in range(3):
    print(f"\n=== Step {step + 1} ===")
    
    # PyTorch optimizer
    pytorch_opt.zero_grad()
    out_pytorch = model_pytorch(x)
    loss_pytorch = ((out_pytorch - y) ** 2).mean()
    loss_pytorch.backward()
    
    # Print gradient and momentum state BEFORE step
    param_pytorch = list(model_pytorch.parameters())[0]
    grad_pytorch = param_pytorch.grad.data
    print(f"PyTorch gradient norm: {grad_pytorch.norm().item():.6f}")
    
    # Check momentum buffer
    state_pytorch = pytorch_opt.state[param_pytorch]
    if 'momentum_buffer' in state_pytorch:
        momentum_pytorch = state_pytorch['momentum_buffer']
        print(f"PyTorch momentum norm BEFORE step: {momentum_pytorch.norm().item():.6f}")
    else:
        print("PyTorch momentum: not initialized yet")
    
    pytorch_opt.step()
    
    # Check momentum buffer AFTER step
    if 'momentum_buffer' in state_pytorch:
        momentum_pytorch = state_pytorch['momentum_buffer']
        print(f"PyTorch momentum norm AFTER step: {momentum_pytorch.norm().item():.6f}")
    
    # Custom optimizer
    custom_opt.zero_grad()
    out_custom = model_custom(x)
    loss_custom = ((out_custom - y) ** 2).mean()
    loss_custom.backward()
    
    # Print gradient
    param_custom = list(model_custom.parameters())[0]
    grad_custom = param_custom.grad.data
    print(f"Custom gradient norm: {grad_custom.norm().item():.6f}")
    
    # Check velocity state
    key = (0, 0)  # First group, first parameter
    if key in custom_opt.custom_opts:
        v_custom = custom_opt.custom_opts[key].v
        if v_custom is not None:
            v_norm = np.linalg.norm(v_custom)
            print(f"Custom velocity norm BEFORE step: {v_norm:.6f}")
    else:
        print("Custom velocity: not initialized yet")
    
    custom_opt.step()
    
    # Check velocity AFTER step
    if key in custom_opt.custom_opts:
        v_custom = custom_opt.custom_opts[key].v
        if v_custom is not None:
            v_norm = np.linalg.norm(v_custom)
            print(f"Custom velocity norm AFTER step: {v_norm:.6f}")
    
    # Compare weights
    weight_diff = torch.abs(param_pytorch.data - param_custom.data).max().item()
    print(f"Weight difference after step: {weight_diff:.6e}")

print("\n=== Final Comparison ===")
print(f"Final weight difference: {torch.abs(model_pytorch.weight - model_custom.weight).max().item():.6e}")
print(f"PyTorch final weight norm: {model_pytorch.weight.norm().item():.6f}")
print(f"Custom final weight norm: {model_custom.weight.norm().item():.6f}")
