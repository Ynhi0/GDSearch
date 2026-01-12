"""Test config loader functionality."""

from src.utils.config_loader import load_optimizer_config, load_experiment_config

print("="*80)
print("CONFIG LOADER TESTS")
print("="*80)

# Test 1: Load Adam config
print("\n[TEST 1] Load Adam optimizer config for ResNet CIFAR-10")
adam_cfg = load_optimizer_config('benchmark_hyperparameters', 'resnet_cifar10', 'Adam')
print(f"[OK] Loaded: lr={adam_cfg['lr']}, beta1={adam_cfg['beta1']}, weight_decay={adam_cfg['weight_decay']}")

# Test 2: Load SGD config
print("\n[TEST 2] Load SGD optimizer config for 2D optimization")
sgd_cfg = load_optimizer_config('benchmark_hyperparameters', '2d_optimization', 'SGD')
print(f"[OK] Loaded: lr={sgd_cfg['lr']}")

# Test 3: Load full experiment config
print("\n[TEST 3] Load full experiment config for ResNet CIFAR-10")
exp_cfg = load_experiment_config('benchmark_hyperparameters', 'resnet_cifar10')
print(f"[OK] Loaded: epochs={exp_cfg['epochs']}, batch_size={exp_cfg['batch_size']}")
print(f"[OK] Available optimizers: {list(exp_cfg['optimizers'].keys())}")

# Test 4: Use config with optimizer
print("\n[TEST 4] Create optimizer from config")
from src.core.optimizers import Adam
adam = Adam(**adam_cfg)
print(f"[OK] Created: {adam.name}")

print("\n" + "="*80)
print("ALL CONFIG LOADER TESTS PASSED")
print("="*80)
