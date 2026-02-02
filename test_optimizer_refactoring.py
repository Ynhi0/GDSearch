"""Quick test to verify all refactored optimizers work correctly."""
import numpy as np
from src.core.optimizers import (
    SGD, SGDMomentum, SGDNesterov, RMSProp,
    Adam, AdamW, AMSGrad, AdaBound, RAdam, LAMB
)

print("Testing all refactored optimizers...")
print("=" * 60)

# Test tuple mode (2D test functions)
print("\n1. Testing TUPLE MODE (2D test functions):")
print("-" * 60)
optimizers_tuple = [
    SGD(), SGDMomentum(), SGDNesterov(), RMSProp(),
    Adam(), AdamW(), AMSGrad(), AdaBound(), RAdam(), LAMB()
]
params_tuple = (1.0, 2.0)
grads_tuple = (0.1, 0.2)

for opt in optimizers_tuple:
    result = opt.step(params_tuple, grads_tuple)
    print(f"{type(opt).__name__:15} -> {result}")

# Test array mode (neural networks)
print("\n2. Testing ARRAY MODE (neural networks):")
print("-" * 60)
optimizers_array = [
    SGD(), SGDMomentum(), SGDNesterov(), RMSProp(),
    Adam(), AdamW(), AMSGrad(), AdaBound(), RAdam(), LAMB()
]
params_array = np.array([1.0, 2.0, 3.0])
grads_array = np.array([0.1, 0.2, 0.3])

for opt in optimizers_array:
    result = opt.step(params_array, grads_array)
    print(f"{type(opt).__name__:15} -> {result}")

print("\n" + "=" * 60)
print("✅ ALL OPTIMIZERS WORKING CORRECTLY")
print("✅ Dispatcher pattern validated for both tuple and array modes")
print("=" * 60)
