# Label Smoothing, AMP, and EMA - Implementation Notes

## Overview
This document clarifies how advanced training features (label smoothing, AMP, EMA) relate to the proposal's baseline requirements.

## Proposal Compliance ✅

### Baseline (Required)
- **Convergence rate study**: Use `label_smoothing=0.0` for strict theory-vs-practice comparisons
- **Deterministic conditions**: Disable augmentation for optimization analysis (`augment=False` in data loaders)
- **Reproducibility**: Multi-seed experiments with statistical tests (paired t-tests, effect sizes, power analysis)

### Extensions (Allowed)
- **Label Smoothing**: Optional regularization technique (typical: 0.05-0.1)
- **AMP (Automatic Mixed Precision)**: Optional performance optimization for GPU training
- **EMA (Exponential Moving Average)**: Optional model averaging for improved generalization

## Usage Guidelines

### For Theory-Practice Validation (Baseline)
```python
config = {
    'model': 'SimpleMLP',
    'dataset': 'MNIST',
    'optimizer': 'Adam',
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 128,
    'seed': 42,
    'label_smoothing': 0.0,  # Required for baseline
    'use_amp': False,
    'use_ema': False
}
```

### For Generalization Study (Extension)
```python
config = {
    'model': 'SimpleMLP',
    'dataset': 'MNIST',
    'optimizer': 'Adam',
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 128,
    'seed': 42,
    'label_smoothing': 0.1,  # Extension: improves calibration
    'use_amp': True,         # Extension: speeds up training
    'use_ema': True,         # Extension: stabilizes evaluation
    'ema_decay': 0.9999
}
```

## Label Smoothing Entropy Floor ⚠️

When using label smoothing, loss has a non-zero minimum (entropy floor):

```python
from src.core.training_utils import LabelSmoothingCrossEntropy

loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
entropy_floor = loss_fn.get_entropy_floor(num_classes=10)
# For CIFAR10/MNIST: entropy_floor ≈ 0.54
```

**Implication**: When plotting loss curves for convergence analysis:
- Subtract the entropy floor: `adjusted_loss = raw_loss - entropy_floor`
- Or compare to the floor explicitly in plots
- Report both raw and adjusted losses in tables

## Running Ablations

Use the provided config for multi-seed paired comparison:

```bash
# Label smoothing ablation (5 seeds, baseline vs smoothing=0.1)
python src/experiments/run_full_analysis.py --config configs/label_smoothing_ablation.json

# Statistical analysis automatically computes:
# - Paired t-tests (same seeds)
# - Effect sizes (Cohen's d)
# - Power analysis
# - Confidence intervals
```

## Integration Tests

Run tests to verify features work end-to-end:

```bash
# Test label smoothing propagation
pytest tests/test_integration_label_smoothing.py::TestLabelSmoothingIntegration -v

# Test AMP propagation (requires CUDA)
pytest tests/test_integration_label_smoothing.py::TestAMPIntegration -v

# Test EMA propagation
pytest tests/test_integration_label_smoothing.py::TestEMAIntegration -v

# Test all features together
pytest tests/test_integration_label_smoothing.py::TestCombinedFeatures -v
```

## Reporting Guidelines

When reporting results with extensions:

1. **Always compare to baseline**: Run matched experiments with and without the extension
2. **Use paired tests**: Same seeds for fair comparison
3. **Report effect sizes**: Not just p-values
4. **Document entropy floor**: When using label smoothing, report the floor value
5. **Specify in captions**: "Extension ablation: results with label_smoothing=0.1"

## Implementation Details

### Files Modified (AUDIT FIX)
- `src/experiments/run_nn_experiment.py`: 
  - Added `label_smoothing`, `use_amp`, `use_ema` config support
  - Wire flags to `get_loss_function()`, `AMPWrapper`, `ModelEMA`
- `configs/label_smoothing_ablation.json`: Multi-seed ablation config
- `tests/test_integration_label_smoothing.py`: End-to-end integration tests

### Backward Compatibility
- Default behavior unchanged: `label_smoothing=0.0`, `use_amp=False`, `use_ema=False`
- Existing configs and experiments continue to work
- Extensions are opt-in via config keys

---

**Last Updated**: January 15, 2026  
**Audit Fix**: Ensures CLI/config flags propagate to canonical training pipeline
