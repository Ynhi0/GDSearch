"""
Diagnostic script to identify the exact difference between 
PyTorch Adam and custom Adam implementation with high learning rate.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.optimizers import Adam as CustomAdam
from src.core.pytorch_optimizers import AdamWrapper

# Test with lr=0.1 (the failing case)
lr = 0.1
print(f"Testing with lr={lr}")

# Create identical models
torch.manual_seed(42)
model_pytorch = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

torch.manual_seed(42)
model_custom = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Verify weights are identical
max_diff = 0.0
for p1, p2 in zip(model_pytorch.parameters(), model_custom.parameters()):
    max_diff = max(max_diff, torch.abs(p1 - p2).max().item())
print(f"Initial weight difference: {max_diff:.6e}")

# Create optimizers
pytorch_opt = torch.optim.Adam(model_pytorch.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
custom_opt = AdamWrapper(model_custom.parameters(), lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Create test data
torch.manual_seed(123)
x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))
criterion = nn.CrossEntropyLoss()

# Do 10 steps
for step in range(10):
    # PyTorch optimizer
    pytorch_opt.zero_grad()
    out_pytorch = model_pytorch(x)
    loss_pytorch = criterion(out_pytorch, y)
    loss_pytorch.backward()
    pytorch_opt.step()
    
    # Custom optimizer  
    custom_opt.zero_grad()
    out_custom = model_custom(x)
    loss_custom = criterion(out_custom, y)
    loss_custom.backward()
    custom_opt.step()
    
    # Compare weights
    max_diff = 0.0
    for p1, p2 in zip(model_pytorch.parameters(), model_custom.parameters()):
        diff = torch.abs(p1 - p2).max().item()
        max_diff = max(max_diff, diff)
    
    print(f"Step {step+1}: max_diff={max_diff:.6e}, loss_pytorch={loss_pytorch.item():.4f}, loss_custom={loss_custom.item():.4f}")

print(f"\nFinal max difference: {max_diff:.6e}")
print(f"Tolerance: 1e-6")
print(f"Exceeds tolerance: {max_diff > 1e-6}")
