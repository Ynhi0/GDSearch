"""
Quick local verification that ResNet-18 code compiles and runs.
This does NOT train the model, just verifies the code works.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from src.core.models import ResNet18, BasicBlock
from src.core.pytorch_optimizers import AdamWrapper

print("=" * 60)
print("ResNet-18 Local Verification")
print("=" * 60)
print()

# Test 1: Model creation
print("Test 1: Creating ResNet-18...")
model = ResNet18(num_classes=10)
num_params = model.get_num_parameters()
print(f"✓ Model created: {num_params:,} parameters")
assert num_params > 11_000_000, "Parameter count seems wrong"
print()

# Test 2: Forward pass
print("Test 2: Forward pass...")
x = torch.randn(4, 3, 32, 32)
output = model(x)
print(f"✓ Input shape: {x.shape}")
print(f"✓ Output shape: {output.shape}")
assert output.shape == (4, 10), "Output shape incorrect"
print()

# Test 3: Backward pass
print("Test 3: Backward pass...")
target = torch.randint(0, 10, (4,))
loss = nn.functional.cross_entropy(output, target)
loss.backward()
print(f"✓ Loss: {loss.item():.4f}")
print(f"✓ Gradients computed")
print()

# Test 4: Custom optimizer
print("Test 4: Custom Adam optimizer...")
optimizer = AdamWrapper(model.parameters(), lr=0.001)
optimizer.zero_grad()
print("✓ Optimizer created")

# Test 5: Training step
print("Test 5: Training step...")
x = torch.randn(4, 3, 32, 32)
y = torch.randint(0, 10, (4,))
output = model(x)
loss = nn.functional.cross_entropy(output, y)
loss.backward()
optimizer.step()
print(f"✓ Training step completed")
print(f"✓ Loss: {loss.item():.4f}")
print()

# Test 6: Residual connections
print("Test 6: Residual connections...")
has_basic_blocks = any(isinstance(m, BasicBlock) for m in model.modules())
print(f"✓ Contains BasicBlock: {has_basic_blocks}")
assert has_basic_blocks, "Should contain BasicBlock modules"
print()

# Summary
print("=" * 60)
print("✅ All local tests passed!")
print("=" * 60)
print()
print("Ready to run on Kaggle:")
print("  1. Go to kaggle/INSTRUCTIONS.md")
print("  2. Follow step-by-step instructions")
print("  3. Copy resnet18_cifar10.py to Kaggle")
print("  4. Enable GPU and run")
print("  5. Share results back")
print()
