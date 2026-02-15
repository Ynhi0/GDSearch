# Code Organization Quick Reference

**Quick guide for using new code organization utilities**

---

## 🚀 Quick Start

### 1. Standard Training Loop

Replace 100+ lines of training code with 5 lines:

```python
from src.experiments.training_loops import standard_classification_loop, TrainingConfig

config = TrainingConfig(epochs=50, device=device, patience=10)
results = standard_classification_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    config=config
)

print(f"Best val accuracy: {results.best_val_acc:.2f}%")
```

### 2. Create Optimizer (Factory Pattern)

Replace 15+ if/elif cases with 1 line:

```python
from src.core.optimizer_factory import OptimizerFactory

# Simple creation
optimizer = OptimizerFactory.create('Adam', model.parameters(), lr=0.001)

# From config
opt_config = {'name': 'SGD', 'lr': 0.1, 'momentum': 0.9}
optimizer = OptimizerFactory.create_from_config(model.parameters(), opt_config)
```

### 3. Create Model (Factory Pattern)

```python
from src.core.model_factory import create_model_for_dataset

# Auto-configure for dataset (num_classes, input_channels, etc.)
model = create_model_for_dataset('SimpleCNN', 'mnist')

# Or explicit
from src.core.model_factory import ModelFactory
model = ModelFactory.create('ResNet18', num_classes=10)
```

### 4. Load Configuration

```python
from src.core.config_loader import load_and_validate_config

# Load and validate in one step
config = load_and_validate_config('configs/nn_tuning.json')

# Or build programmatically
from src.core.config_loader import ConfigLoader
config = ConfigLoader.create_experiment_config(
    dataset='mnist',
    optimizers=['SGD', 'Adam', 'AdamW'],
    seeds=[42, 123, 456],
    epochs=50
)
```

### 5. Use Named Constants

```python
from src.utils.constants import (
    ADAM_DEFAULT_LR,
    DEFAULT_BATCH_SIZE_MNIST,
    MAX_SAFE_LOSS
)

# Self-documenting code
lr = ADAM_DEFAULT_LR  # Standard Adam default (Kingma & Ba, 2015)
batch_size = DEFAULT_BATCH_SIZE_MNIST  # Optimized for T4 GPU
if loss > MAX_SAFE_LOSS:
    logging.error("Numerical instability!")
```

---

## 📚 Common Patterns

### Pattern 1: Complete MNIST Experiment (5 lines)

```python
from src.experiments.training_loops import standard_classification_loop, TrainingConfig
from src.core.optimizer_factory import OptimizerFactory
from src.core.model_factory import create_model_for_dataset
from src.utils.constants import ADAM_DEFAULT_LR, DEFAULT_BATCH_SIZE_MNIST

model = create_model_for_dataset('SimpleCNN', 'mnist')
optimizer = OptimizerFactory.create('Adam', model.parameters(), lr=ADAM_DEFAULT_LR)
config = TrainingConfig(epochs=50, device=torch.device('cuda'))
results = standard_classification_loop(model, train_loader, val_loader, optimizer, criterion, config)
print(f"Final accuracy: {results.final_test_acc:.2f}%")
```

### Pattern 2: Multi-Optimizer Comparison

```python
from src.core.optimizer_factory import OptimizerFactory

optimizers_config = [
    {'name': 'SGD', 'lr': 0.1, 'momentum': 0.9},
    {'name': 'Adam', 'lr': 0.001},
    {'name': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01},
]

results = {}
for opt_config in optimizers_config:
    model = create_model_for_dataset('ResNet18', 'cifar10')
    optimizer = OptimizerFactory.create_from_config(model.parameters(), opt_config)
    result = standard_classification_loop(model, train_loader, val_loader, optimizer, criterion, config)
    results[opt_config['name']] = result.best_val_acc
```

### Pattern 3: Custom Optimizer Registration

```python
from src.core.optimizer_factory import OptimizerFactory
from my_module import MyCustomOptimizer

OptimizerFactory.register(
    'MyOptimizer',
    MyCustomOptimizer,
    default_hyperparams={'lr': 0.01, 'beta': 0.9}
)

# Now use like any other optimizer
optimizer = OptimizerFactory.create('MyOptimizer', model.parameters())
```

---

## 🔧 Migration Examples

### Before/After: Training Loop

**BEFORE (100+ lines):**
```python
for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(targets).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / len(train_dataset)
    
    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            # ... validation logic
    
    # Learning rate scheduling
    scheduler.step()
    
    # Best model tracking
    if val_acc > best_val_acc:
        # ... save best model
    
    # Early stopping
    # ... early stopping logic
    
    # Checkpointing
    # ... checkpoint saving
```

**AFTER (5 lines):**
```python
from src.experiments.training_loops import standard_classification_loop, TrainingConfig

config = TrainingConfig(epochs=epochs, device=device, patience=10)
results = standard_classification_loop(
    model, train_loader, val_loader, optimizer, criterion, config,
    scheduler=scheduler, checkpoint_manager=checkpoint_manager
)
```

### Before/After: Optimizer Creation

**BEFORE (20+ lines):**
```python
if opt_name == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif opt_name == 'SGD_Momentum':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
elif opt_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
elif opt_name == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
elif opt_name == 'AMSGrad':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
# ... 10+ more cases
else:
    raise ValueError(f"Unknown optimizer: {opt_name}")
```

**AFTER (1 line):**
```python
optimizer = OptimizerFactory.create(opt_name, model.parameters(), lr=lr)
```

---

## 📖 API Reference

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    epochs: int                              # Required
    device: torch.device                     # Required
    patience: int = 10                       # Early stopping patience
    grad_clip_norm: Optional[float] = None   # Gradient clipping
    use_amp: bool = False                    # Mixed precision
    log_interval: int = 10                   # Logging frequency
    compute_grad_noise_every: int = 0        # Gradient noise estimation
    checkpoint_every: int = 1                # Checkpoint frequency
```

### TrainingResults

```python
@dataclass
class TrainingResults:
    history: List[Dict]                      # Per-epoch metrics
    best_val_acc: float                      # Best validation accuracy
    best_val_loss: float                     # Best validation loss
    best_model_state: Optional[Dict]         # Best model weights
    final_test_acc: float                    # Final test accuracy
    total_training_time: float               # Training duration
    early_stopped_at_epoch: Optional[int]    # Early stopping epoch
```

### OptimizerFactory Methods

```python
OptimizerFactory.create(name, params, lr=None, **kwargs)
OptimizerFactory.create_from_config(params, config_dict)
OptimizerFactory.register(name, optimizer_class, defaults)
OptimizerFactory.list_optimizers()  # Get all available optimizers
OptimizerFactory.is_registered(name)
```

### ConfigLoader Methods

```python
ConfigLoader.load_experiment_config(path)
ConfigLoader.merge_configs(base, override, deep=True)
ConfigLoader.apply_defaults(config, defaults)
ConfigLoader.validate_required_fields(config, required)
ConfigLoader.get_dataset_defaults(dataset_name)
ConfigLoader.save_config(config, output_path)
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Wrong DataLoader for Tuning

```python
# ❌ WRONG - Using test loader for validation
results = standard_classification_loop(
    model, train_loader, test_loader, optimizer, criterion, config
)

# ✅ CORRECT - Use validation loader
results = standard_classification_loop(
    model, train_loader, val_loader, optimizer, criterion, config
)
```

### Pitfall 2: Forgetting Device

```python
# ❌ WRONG - Model on GPU, config says CPU
model = model.to('cuda')
config = TrainingConfig(epochs=50, device=torch.device('cpu'))

# ✅ CORRECT - Consistent device
device = torch.device('cuda')
model = model.to(device)
config = TrainingConfig(epochs=50, device=device)
```

### Pitfall 3: Mutating Config

```python
# ❌ WRONG - Mutating shared config
base_config = ConfigLoader.get_dataset_defaults('mnist')
config1 = base_config
config1['epochs'] = 100  # Mutates base_config!

# ✅ CORRECT - Use merge/apply_defaults (creates copy)
base_config = ConfigLoader.get_dataset_defaults('mnist')
config1 = ConfigLoader.apply_defaults({'epochs': 100}, base_config)
```

---

## 🧪 Testing

### Test Training Loop

```python
def test_training_loop():
    model = SimpleCNN(num_classes=10)
    # ... create mock data loaders
    config = TrainingConfig(epochs=3, device=torch.device('cpu'))
    results = standard_classification_loop(
        model, train_loader, val_loader, optimizer, criterion, config
    )
    assert len(results.history) <= 3
    assert results.best_val_acc >= 0
```

### Test Optimizer Factory

```python
def test_optimizer_factory():
    model = nn.Linear(10, 2)
    optimizer = OptimizerFactory.create('Adam', model.parameters(), lr=0.001)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == 0.001
```

---

## 📝 Best Practices

1. **Use named constants** for all numeric values
2. **Always validate configs** before long experiments
3. **Use factories** instead of if/elif chains
4. **Prefer TrainingConfig** over passing many arguments
5. **Check results.run_tainted** after OOM recovery
6. **Use val_loader** for early stopping (never test_loader)
7. **Save configs** alongside results for reproducibility

---

## 🔗 Related Documentation

- `CODE_ORGANIZATION_IMPROVEMENTS.md` - Full implementation details
- `MASTER_FIX_TRACKER.md` - Progress tracking
- `EXPERIMENT_EXECUTION_GUIDE.md` - Experiment workflows
- `.github/copilot-instructions.md` - Development guidelines

---

**Last Updated:** February 2, 2026
