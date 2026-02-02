# Quick Reference: Using the New Audit Fixes

This is a quick reference for using the newly implemented audit fixes in your experiments.

## 🎯 Multi-Seed Experiments (C1)

### Command Line
```bash
# Recommended: Use --seeds with comma-separated values
python kaggle/resnet18_cifar10.py --seeds 42,123,456

# Old way (deprecated, shows warning)
python kaggle/resnet18_cifar10.py --seed 42
```

### In Your Scripts
```python
parser.add_argument('--seeds', type=str, default='42,123,456',
                   help='Comma-separated seeds (e.g., "42,123,456")')
parser.add_argument('--seed', type=int, default=None,
                   help='DEPRECATED: Use --seeds instead')

args = parser.parse_args()

# Handle seed parameters
if args.seed is not None:
    warnings.warn("--seed is deprecated. Use --seeds...", DeprecationWarning)
    seeds = [args.seed]
else:
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

# Run experiments
for seed in seeds:
    set_seed(seed)
    result = run_experiment(seed=seed)
```

---

## 📝 Result Filename Generation (H1)

### Generating Filenames
```python
from src.utils.result_filename import generate_result_filename

# Generate canonical filename
filename = generate_result_filename(
    model="ResNet18",
    dataset="CIFAR10",
    optimizer="Adam",
    lr=0.001,
    seed=42
)
# Result: "NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv"

# With optional tag
filename = generate_result_filename(
    model="ResNet18",
    dataset="CIFAR10",
    optimizer="Adam",
    lr=0.001,
    seed=42,
    tag="ablation"
)
# Result: "NN_ResNet18_CIFAR10_Adam_lr0.001_seed42_ablation.csv"
```

### Parsing Filenames
```python
from src.utils.result_filename import parse_result_filename

components = parse_result_filename("NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv")
# Returns: {'model': 'ResNet18', 'dataset': 'CIFAR10', 
#           'optimizer': 'Adam', 'lr': 0.001, 'seed': 42, 'tag': None}
```

---

## ✅ Config Validation (H2)

### Validating Configs
```python
from src.core.config_loader import validate_config_compatibility

# Valid combination - no error
config = {"dataset": "CIFAR10", "model": "ResNet18"}
validate_config_compatibility(config)  # OK

# Invalid combination - raises ValueError
config = {"dataset": "CIFAR10", "model": "SimpleLSTM"}
validate_config_compatibility(config)
# ValueError: Invalid model 'SimpleLSTM' for dataset 'CIFAR10'
```

### Compatibility Matrix
```python
MNIST       → SimpleMLP, SimpleCNN
FashionMNIST → SimpleMLP, SimpleCNN
CIFAR10     → SimpleCNN, ConvNet, ResNet18
CIFAR100    → ConvNet, ResNet18
IMDB        → SimpleRNN, SimpleLSTM, BiLSTM, TextCNN
PathMNIST   → SimpleCNN, ConvNet
```

---

## 🔬 Optuna Validation (H3)

### With Validation Loader (Recommended)
```python
from src.core.optuna_tuner import OptunaHyperparameterTuner
from src.core.loader_validation import create_validated_loaders

# Create validated loaders
train_loader, val_loader, test_loader = create_validated_loaders(
    get_mnist_loaders,
    val_split=0.15,
    batch_size=128
)

# Use in tuning (recommended)
tuner = OptunaHyperparameterTuner(objective_fn, ...)
results = tuner.optimize(
    n_trials=100,
    val_loader=val_loader,  # Prevents test set leakage
    test_dataset=test_loader.dataset  # Optional but recommended
)
```

### Without Validation Loader (Grace Period)
```python
# Works but shows FutureWarning
results = tuner.optimize(n_trials=100)
# Warning: "will REQUIRE validation in version 2.0"
```

---

## 🛡️ AMSGrad Safety (C2)

### What Changed
Parameter shape changes during training now **abort immediately** with a clear error:

```python
# Before (v1.x - buggy):
# Shape change → logging.error() → continue (corrupted state) ❌

# Now (fixed):
# Shape change → RuntimeError → abort immediately ✅
```

### What This Means
If you see this error:
```
RuntimeError: AMSGrad CRITICAL ERROR: Parameter shape changed from (100,) to (200,).
Shape changes violate AMSGrad's convergence guarantees...
```

**Fix your model/data pipeline** - don't try to suppress this error!

### Common Causes
1. **Dynamic reshaping in model**:
   ```python
   # BAD:
   if self.training:
       x = x.reshape(-1, 128)  # ❌ Shape changes
   
   # GOOD:
   x = x.view(x.size(0), -1)  # ✅ Batch-aware
   ```

2. **Inconsistent data loader**:
   ```python
   # Validate your dataset
   for batch in train_loader:
       assert batch[0].shape[1:] == expected_shape
   ```

---

## 🔢 Test Function Constants (L2)

### Using Constants
```python
from src.core.test_functions import (
    ROSENBROCK_DEFAULT_A,
    ROSENBROCK_DEFAULT_B,
    Rosenbrock
)

# Use defaults (recommended)
rosenbrock = Rosenbrock()

# Custom parameters (still supported)
rosenbrock = Rosenbrock(a=1.0, b=50.0)

# Access constants
print(f"Default a: {ROSENBROCK_DEFAULT_A}")  # 1.0
print(f"Default b: {ROSENBROCK_DEFAULT_B}")  # 100.0
```

### Available Constants
```python
ROSENBROCK_DEFAULT_A = 1.0
ROSENBROCK_DEFAULT_B = 100.0
QUADRATIC_DEFAULT_KAPPA = 100
ACKLEY_DEFAULT_A = 20.0
ACKLEY_DEFAULT_B = 0.2
ACKLEY_DEFAULT_C = 2 * np.pi
RASTRIGIN_DEFAULT_A = 10
```

---

## 📋 Quick Checklist for New Experiments

- [ ] Use `--seeds 42,123,456` (not `--seed 42`)
- [ ] Generate result filenames with `generate_result_filename()`
- [ ] Validate config with `validate_config_compatibility()`
- [ ] Provide `val_loader` to Optuna tuning
- [ ] Ensure model doesn't change parameter shapes
- [ ] Use test function constants instead of magic numbers

---

## 🔍 Verification Commands

### Test All Fixes
```bash
python scripts/test_audit_fixes.py
```

### Test Imports
```bash
python -c "from src.utils.result_filename import generate_result_filename; \
           from src.core.config_loader import validate_config_compatibility; \
           print('✓ Imports work')"
```

### Check Deprecation Warnings
```bash
# Run with warnings as errors to find deprecated usage
python -W error::DeprecationWarning your_script.py
```

---

## 📚 Documentation

- **Detailed Guide**: `docs/MIGRATION_v2.md`
- **Implementation Summary**: `docs/AUDIT_IMPLEMENTATION_SUMMARY.md`
- **Complete Report**: `AUDIT_IMPLEMENTATION_COMPLETE.md`

---

## ❓ Common Questions

### Q: Do I need to update my old scripts?
**A**: Not immediately. Old usage still works with deprecation warnings. You have until v2.0 (Q2 2026) to migrate.

### Q: What happens if I ignore the warnings?
**A**: Your code will break in v2.0. Use the grace period to migrate.

### Q: How do I migrate my existing result files?
**A**: See `docs/MIGRATION_v2.md` for renaming strategies.

### Q: Can I disable the compatibility checks?
**A**: Not recommended. They prevent common mistakes that waste compute time.

---

*Quick Reference v1.0 - December 2025*
