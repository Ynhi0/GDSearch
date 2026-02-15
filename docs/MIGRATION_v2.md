# Migration Guide to GDSearch v2.0

This document outlines the breaking changes planned for GDSearch v2.0 and provides migration paths for deprecated features.

## Timeline

- **Current Version**: 1.x (with deprecation warnings)
- **v2.0 Release**: Planned for Q2 2026
- **Migration Period**: 6 months (current - Q2 2026)

## Breaking Changes & Migration Paths

### 1. **Multi-Seed Experiments (Priority: CRITICAL)**

#### What's Changing
- Single `--seed` parameter will be removed
- `--seeds` (plural, comma-separated) becomes mandatory for all experiments

#### Current Behavior (v1.x with warnings)
```bash
# OLD (deprecated, will show warning):
python kaggle/resnet18_cifar10.py --seed 42

# NEW (recommended):
python kaggle/resnet18_cifar10.py --seeds 42,123,456
```

#### Migration Path
1. **Find all uses of `--seed` in your scripts**:
   ```bash
   grep -r "--seed" your_experiment_scripts/
   ```

2. **Replace with `--seeds`**:
   ```python
   # OLD:
   parser.add_argument('--seed', type=int, default=42)
   
   # NEW:
   parser.add_argument('--seeds', type=str, default='42,123,456')
   seeds = [int(s.strip()) for s in args.seeds.split(',')]
   ```

3. **Update experiment loops**:
   ```python
   # OLD:
   set_seed(args.seed)
   result = run_experiment(seed=args.seed)
   
   # NEW:
   for seed in seeds:
       set_seed(seed)
       result = run_experiment(seed=seed)
       # Save result with seed in filename
   ```

#### Affected Files
- `kaggle/resnet18_cifar10.py` ✅ Updated
- `scripts/train_lstm_imdb.py` ✅ Updated
- All custom experiment scripts using `--seed`

---

### 2. **Optuna Validation Enforcement (Priority: HIGH)**

#### What's Changing
- `OptunaHyperparameterTuner.optimize()` will **require** `val_loader` parameter
- `enforce_validation` default changes from `None` (grace period) to `True` (strict)

#### Current Behavior (v1.x with warnings)
```python
# OLD (will show FutureWarning):
tuner.optimize(n_trials=100)

# NEW (required in v2.0):
tuner.optimize(n_trials=100, val_loader=val_loader)
```

#### Migration Path
1. **Update all Optuna tuning calls**:
   ```python
   from src.core.loader_validation import create_validated_loaders
   
   # Create validated loaders
   train_loader, val_loader, test_loader = create_validated_loaders(
       get_mnist_loaders,
       val_split=0.15,
       batch_size=128
   )
   
   # Use in tuning
   results = tuner.optimize(
       n_trials=100,
       val_loader=val_loader,  # REQUIRED in v2.0
       test_dataset=test_loader.dataset  # Recommended for stronger checks
   )
   ```

2. **If you MUST use test set for tuning** (NOT RECOMMENDED):
   ```python
   # Explicitly disable validation (invalidates research claims)
   results = tuner.optimize(
       n_trials=100,
       enforce_validation=False  # Will be removed in v2.0
   )
   ```

#### Rationale
Prevents test set leakage during hyperparameter tuning, which invalidates research results. This is a **scientific integrity** requirement.

#### Affected Files
- All scripts using `OptunaHyperparameterTuner`
- See `src/core/optuna_tuner.py` for examples

---

### 3. **Result Filename Format (Priority: HIGH)**

#### What's Changing
- Legacy filename formats will no longer be parsed
- Canonical format becomes mandatory: `NN_<Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>[_tag].csv`

#### Current Behavior (v1.x with warnings)
```python
# OLD (deprecated, will show warning):
"NN_SimpleMNIST_Adam_lr0.001_seed42.csv"  # SimpleDirect format
"ResNet18_CIFAR10_Adam_lr0.001_seed42.csv"  # Missing NN_ prefix

# NEW (canonical):
"NN_SimpleMLP_MNIST_Adam_lr0.001_seed42.csv"
"NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv"
```

#### Migration Path
1. **Use centralized filename generator**:
   ```python
   from src.utils.result_filename import generate_result_filename
   
   # Generate canonical filename
   filename = generate_result_filename(
       model="ResNet18",
       dataset="CIFAR10",
       optimizer="Adam",
       lr=0.001,
       seed=42,
       tag=None  # Optional
   )
   # Result: "NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv"
   ```

2. **Rename existing result files**:
   ```python
   from src.utils.result_filename import parse_result_filename, generate_result_filename
   import os
   
   # Parse old filename
   old_file = "ResNet18_CIFAR10_Adam_lr0.001_seed42.csv"
   components = parse_result_filename(old_file)
   
   # Generate new filename
   new_file = generate_result_filename(**components)
   
   # Rename
   os.rename(f"results/{old_file}", f"results/{new_file}")
   ```

3. **Update plotting scripts**:
   ```python
   # OLD:
   filename = f"{model}_{dataset}_{optimizer}_lr{lr}_seed{seed}.csv"
   
   # NEW:
   from src.utils.result_filename import generate_result_filename
   filename = generate_result_filename(model, dataset, optimizer, lr, seed)
   ```

#### Affected Files
- `src/experiments/run_experiment.py`
- `scripts/run_final_benchmarks.py`
- All custom experiment and analysis scripts

---

### 4. **Config Validation (Priority: MEDIUM)**

#### What's Changing
- `load_and_validate_config()` will automatically check dataset-model compatibility
- Invalid combinations (e.g., LSTM on CIFAR10) will raise errors instead of warnings

#### Current Behavior (v1.x)
```python
# OLD (may work but produce nonsensical results):
config = {
    "dataset": "CIFAR10",
    "model": "SimpleLSTM"  # Text model on image data!
}

# NEW (will raise ValueError in v2.0):
from src.core.config_loader import load_and_validate_config
config = load_and_validate_config("configs/invalid_config.json")
# ValueError: Invalid model 'SimpleLSTM' for dataset 'CIFAR10'
```

#### Migration Path
1. **Review all config files**:
   ```bash
   python scripts/validate_configs.py --config configs/*.json
   ```

2. **Fix incompatible combinations**:
   ```json
   // BAD:
   {
     "dataset": "CIFAR10",
     "model": "SimpleLSTM"
   }
   
   // GOOD:
   {
     "dataset": "CIFAR10",
     "model": "ResNet18"
   }
   ```

3. **Add new models to compatibility matrix** (if custom):
   ```python
   # In src/core/config_loader.py
   DATASET_MODEL_COMPATIBILITY = {
       "CIFAR10": ["SimpleCNN", "ConvNet", "ResNet18", "YourCustomCNN"],
       # ...
   }
   ```

#### Compatibility Matrix
```python
DATASET_MODEL_COMPATIBILITY = {
    "MNIST": ["SimpleMLP", "SimpleCNN"],
    "FashionMNIST": ["SimpleMLP", "SimpleCNN"],
    "CIFAR10": ["SimpleCNN", "ConvNet", "ResNet18"],
    "CIFAR100": ["ConvNet", "ResNet18"],
    "IMDB": ["SimpleRNN", "SimpleLSTM", "BiLSTM", "TextCNN"],
    "PathMNIST": ["SimpleCNN", "ConvNet"]
}
```

---

### 5. **AMSGrad Shape Change Handling (Priority: CRITICAL)**

#### What's Changing
- Shape changes during training will **abort** with RuntimeError instead of logging and continuing
- This prevents silent corruption of optimizer state

#### Current Behavior (v1.x)
```python
# OLD (v1.x - would reset state and continue):
# Parameter shape changed → logging.error() → continue training

# NEW (v2.0 - aborts immediately):
# Parameter shape changed → raise RuntimeError
```

#### Migration Path
1. **Fix shape-changing bugs in your model**:
   ```python
   # BAD (shape changes during training):
   class BadModel(nn.Module):
       def forward(self, x):
           if self.training:
               x = x.reshape(-1, 128)  # ❌ Dynamic reshaping
           return self.fc(x)
   
   # GOOD (fixed shape):
   class GoodModel(nn.Module):
       def forward(self, x):
           x = x.view(x.size(0), -1)  # ✅ Batch-aware reshaping
           return self.fc(x)
   ```

2. **Ensure consistent input shapes**:
   ```python
   # Validate dataset returns consistent shapes
   for batch in train_loader:
       assert batch[0].shape[1:] == expected_shape, "Inconsistent input shape!"
   ```

#### Rationale
Shape changes violate AMSGrad's convergence guarantees (Reddi et al., 2018). Continuing training produces scientifically invalid results.

---

## Removed Features in v2.0

### 1. **Legacy Unfair Ablations**
- `--allow-unfair-ablations` flag will be removed
- All ablations will use per-optimizer fair defaults by default

**Migration**: Remove this flag from your scripts. Fair defaults are now mandatory.

### 2. **Inline SAM Implementations**
- Inline `SAMSGD`, `SAMAdam` classes removed from scripts
- Use unified `SAMWrapper` from `src.core.pytorch_optimizers`

**Migration**:
```python
# OLD:
from inline_sam import SAMSGD  # ❌ Removed

# NEW:
from src.core.pytorch_optimizers import SAMWrapper
base_opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = SAMWrapper(base_opt, rho=0.05)
```

---

## New Requirements in v2.0

### 1. **Minimum 3 Seeds for Statistical Validity**
- Experiments with < 3 seeds will show warnings
- Papers/reports should use ≥ 5 seeds for robust results

### 2. **Validation Splits Mandatory for Tuning**
- All hyperparameter tuning must use validation set
- Test set must not be used for any tuning decisions

### 3. **Result Filename Compliance**
- All results must use canonical filename format
- Analysis scripts will reject non-canonical filenames

---

## Automated Migration Tools

### Check Deprecation Warnings
```bash
# Run experiments with Python warnings as errors
python -W error::DeprecationWarning kaggle/resnet18_cifar10.py
```

### Validate Configurations
```bash
# Check all configs for compatibility issues
python scripts/validate_configs.py --config configs/*.json
```

### Rename Result Files
```bash
# Auto-rename legacy result files to canonical format
python scripts/migrate_result_filenames.py --results-dir results/
```

---

## Timeline Summary

| Date | Milestone |
|------|-----------|
| **Dec 2025** | Deprecation warnings added (current) |
| **Jan 2026** | Migration tools released |
| **Mar 2026** | Final migration reminder in release notes |
| **Q2 2026** | **v2.0 Release** - Breaking changes take effect |

---

## Getting Help

- **Documentation**: See `docs/` for updated examples
- **Issues**: Report migration problems on GitHub Issues
- **Questions**: Use GitHub Discussions for migration questions

---

## Version Compatibility

### Backward Compatibility (v1.x)
- All deprecated features still work with warnings
- No breaking changes in v1.x releases
- Gradual migration recommended

### Forward Compatibility (v2.0)
- Code following new patterns will work in both v1.x and v2.0
- Recommended: Update to new patterns now to avoid rush before v2.0

---

## Summary Checklist

Before v2.0 release, ensure:

- [ ] All `--seed` replaced with `--seeds`
- [ ] All Optuna calls include `val_loader`
- [ ] All result filenames use canonical format
- [ ] All configs validated for dataset-model compatibility
- [ ] No dynamic shape changes in models (AMSGrad-safe)
- [ ] Minimum 3 seeds used in experiments
- [ ] Inline SAM implementations replaced with `SAMWrapper`
- [ ] All deprecation warnings resolved

---

**Last Updated**: December 2025  
**Version**: 1.x → 2.0 Migration Guide
