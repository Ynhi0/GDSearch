# Documentation Implementation Roadmap

**Project:** GDSearch Documentation Remediation  
**Start Date:** TBD  
**Duration:** 3 weeks (120 hours)  
**Objective:** Achieve ≥90% documentation coverage and publication-ready standards

---

## Prerequisites

Before starting, install documentation tools:

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install documentation tools
pip install pydocstyle interrogate mypy

# Run baseline measurements
pydocstyle src/ --convention=google > docs/pydocstyle_baseline.txt
interrogate -vv src/ > docs/interrogate_baseline.txt
mypy --strict src/ > docs/mypy_baseline.txt
```

---

## Phase 1: Critical Blockers (Week 1 - 40 hours)

### Task 1.1: Complete Optimizer Documentation (16 hours)

**File:** `src/core/optimizers.py`

**Action Plan:**
For each of the 13 optimizer classes, add complete documentation following this template:

```python
class OptimizerName(Optimizer):
    """
    One-line summary of optimizer purpose.
    
    Longer description explaining:
    - What problem this optimizer solves
    - Key innovation vs alternatives
    - When to use this optimizer
    
    Algorithm:
        1. Step 1 description with mathematical notation
        2. Step 2 description
        3. ...
    
    Mathematical Formulation:
        Equations using LaTeX-style notation
        
    Args:
        param1: Description including valid ranges and defaults
        param2: Description
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid parameters provided
        
    Example:
        >>> # 2D function optimization
        >>> from src.core.test_functions import rosenbrock_gradient
        >>> optimizer = OptimizerName(lr=0.01)
        >>> params = (-1.5, 2.0)
        >>> for i in range(1000):
        >>>     grads = rosenbrock_gradient(params)
        >>>     params = optimizer.step(params, grads)
        >>> print(f"Final params: {params}")
        
        >>> # Neural network training
        >>> from src.core.pytorch_optimizers import OptimizerNameWrapper
        >>> optimizer = OptimizerNameWrapper(model.parameters(), lr=0.001)
        >>> optimizer.zero_grad()
        >>> loss = criterion(model(data), target)
        >>> loss.backward()
        >>> optimizer.step()
    
    Note:
        - Computational complexity: O(?) per step
        - Memory overhead: ?x base optimizer
        - Convergence rate: O(1/k) for convex functions
        - Best suited for: [problem characteristics]
        
    Performance Characteristics:
        - Works best with batch size: [range]
        - Recommended learning rate: [range]
        - Typical hyperparameters: ...
        
    References:
        Author, et al. "Paper Title." Conference/Journal Year.
        https://arxiv.org/abs/xxxx.xxxxx
        
    See Also:
        - RelatedOptimizer: Brief comparison
        - pytorch_optimizers.OptimizerWrapper: PyTorch integration
    """
```

**Specific Tasks:**

1. **SGD** (2 hours)
   - Add Example section
   - Add Note about simplicity vs adaptivity tradeoff
   - Reference: Robbins & Monro (1951)

2. **SGDMomentum** (2 hours)
   - Add algorithm steps
   - Add Example section
   - Reference: Polyak (1964)
   - Note convergence acceleration

3. **SGDNesterov** (2 hours)
   - Add mathematical formulation
   - Add Example section
   - Reference: Nesterov (1983)
   - Compare to standard momentum

4. **RMSProp** (1.5 hours)
   - Add Example section
   - Reference: Hinton's Coursera lecture
   - Note adaptive LR behavior

5. **Adam** (2 hours)
   - Add complete Examples
   - Reference: Kingma & Ba (2015)
   - Note bias correction importance
   - Add See Also: AdamW, AMSGrad

6. **AdamW** (1.5 hours)
   - Add Example section
   - Reference: Loshchilov & Hutter (2019)
   - Explain decoupled weight decay advantage

7. **AMSGrad** (1.5 hours)
   - Add Example section
   - Reference: Reddi et al. (2018)
   - Explain non-increasing step sizes

8. **SAM** (2 hours) - **PRIORITY**
   - Complete Example section
   - Add Note about 2x computational cost
   - Reference: Foret et al. (2021) - COMPLETE
   - Add See Also: ASAM, LookSAM

9. **Lookahead** (1.5 hours)
   - Add algorithm explanation
   - Add Example section
   - Reference: Zhang et al. (2019) - COMPLETE
   - Note slow weights limitation

10. **AdaBound** (1 hour)
    - Complete reference (Luo et al. 2019)
    - Add Example section

11. **RAdam** (1 hour)
    - Format reference properly
    - Add Example section
    - Explain warmup heuristic

12. **LAMB** (1 hour)
    - Add Example section
    - Reference: You et al. (2020)
    - Explain layer-wise adaptation

**Validation:**
```bash
# After completing each optimizer, run:
pydocstyle src/core/optimizers.py --convention=google
# Should show zero errors for completed sections
```

---

### Task 1.2: Create Package README Files (8 hours)

Create the following README files:

#### `src/README.md` (1 hour)

```markdown
# GDSearch Source Code

## Overview
This directory contains the core implementation of the GDSearch optimizer dynamics research platform.

## Package Structure

```
src/
├── core/          # Core algorithms and optimizers
├── experiments/   # Experiment runners and orchestrators
├── utils/         # Shared utilities and helpers
├── analysis/      # Analysis and metrics computation
├── visualization/ # Plotting and visualization
├── runners/       # High-level experiment runners
└── data/          # Data loading and preprocessing
```

## Quick Navigation

- **Implementing a new optimizer?** → See `core/README.md`
- **Running experiments?** → See `experiments/README.md`
- **Adding visualization?** → See `visualization/README.md`
- **Need utilities?** → See `utils/README.md`

## Import Patterns

All imports should be absolute from project root:
```python
from src.core.optimizers import Adam
from src.experiments.training_loops import standard_classification_loop
```

## Dependencies

See `requirements.txt` in project root. Core dependencies:
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- Pandas ≥ 2.0

## Testing

Run tests from project root:
```bash
pytest tests/ -v
```
```

#### `src/core/README.md` (1.5 hours)

```markdown
# Core Algorithms Package

## Purpose
Contains the fundamental building blocks of the GDSearch research platform:
- Optimization algorithms (12 optimizers)
- PyTorch wrappers for optimizers
- Neural network architectures
- Test functions (2D and high-dimensional)
- Training utilities
- Data loading utilities

## Key Components

### Optimizers (`optimizers.py`)
13 optimization algorithms for 2D function optimization:
- `SGD`, `SGDMomentum`, `SGDNesterov`
- `RMSProp`
- `Adam`, `AdamW`, `AMSGrad`
- `SAM` (Sharpness-Aware Minimization)
- `Lookahead`
- `AdaBound`, `RAdam`, `LAMB`

**Usage:**
```python
from src.core.optimizers import Adam
optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
params = (-1.5, 2.0)
grads = compute_gradients(params)
new_params = optimizer.step(params, grads)
```

### PyTorch Wrappers (`pytorch_optimizers.py`)
PyTorch-compatible wrappers for neural network training:
- `SGDWrapper`, `SGDMomentumWrapper`, `AdamWrapper`, etc.
- `SAMWrapper` (requires closure for double backward pass)
- `LookaheadWrapper`

**Usage:**
```python
from src.core.pytorch_optimizers import AdamWrapper
optimizer = AdamWrapper(model.parameters(), lr=0.001)
```

### Models (`models.py`)
Neural network architectures:
- `SimpleMLP` (MNIST)
- `SimpleCNN`, `ConvNet` (CIFAR-10)
- `ResNet18` (18-layer residual network)
- NLP models: `SimpleRNN`, `SimpleLSTM`, `BiLSTM`, `TextCNN`

### Test Functions (`test_functions.py`)
2D and high-dimensional optimization benchmarks:
- `rosenbrock`, `ackley_2d`, `rastrigin`
- `sphere`, `schwefel`, `ill_conditioned_quadratic`
- High-dimensional variants for N-D testing

### Experiment Tracking (`experiment_tracker.py`)
MLflow integration for logging experiments:
```python
from src.core.experiment_tracker import ExperimentTracker
tracker = ExperimentTracker("MyExperiment")
tracker.start_run()
tracker.log_metrics({"loss": 0.5})
tracker.end_run()
```

## Adding a New Optimizer

1. Create class inheriting from `Optimizer` in `optimizers.py`
2. Implement `step()`, `reset()` methods
3. Add complete Google-style docstring with Examples
4. Create PyTorch wrapper in `pytorch_optimizers.py`
5. Add unit tests in `tests/test_optimizers.py`
6. Add to `optimizer_factory.py` registry

## Testing

```bash
# Test optimizers
pytest tests/test_optimizers.py -v

# Test PyTorch wrappers
pytest tests/test_pytorch_optimizers.py -v

# Test models
pytest tests/test_models.py -v
```

## References

See individual optimizer docstrings for paper citations.
```

#### `src/experiments/README.md` (1.5 hours)

```markdown
# Experiments Package

## Purpose
Contains high-level experiment runners and orchestration scripts for systematic benchmarking.

## Main Runners

### `run_nn_experiment.py`
Single neural network training run (MNIST/CIFAR-10).

**Usage:**
```bash
python -m src.experiments.run_nn_experiment \
    --dataset MNIST \
    --model SimpleMLP \
    --optimizer Adam \
    --lr 0.001 \
    --epochs 50 \
    --seed 42
```

### `run_multi_seed.py`
Multi-seed experiments for statistical validity.

**Usage:**
```bash
python -m src.experiments.run_multi_seed \
    --config configs/nn_tuning.json \
    --seeds 42,123,456,789,1011
```

### `run_optimizer_ablation.py`
Fair comparison of optimizers with per-optimizer default LRs.

**Usage:**
```bash
python -m src.experiments.run_optimizer_ablation \
    --dataset MNIST \
    --optimizers SGD,Adam,AdamW \
    --seeds 42,123,456
```

### `run_cifar10.py`
CIFAR-10 benchmarks with ResNet-18.

### `run_transformer_nlp.py`
NLP experiments on IMDB sentiment analysis.

### `training_loops.py`
Standardized training loop implementations (DRY principle).

**Key Function:**
```python
from src.experiments.training_loops import standard_classification_loop

results = standard_classification_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    config=training_config,
    test_loader=test_loader  # Optional
)
```

## Ablation Studies

### `scheduler_ablation.py`
Compare learning rate schedulers.

### `weight_decay_ablation.py`
Analyze weight decay effects.

### `label_noise_ablation.py`
Robustness to label noise.

### `initialization_ablation.py`
Compare weight initialization methods.

## Running Experiments

### Quick Test (Fast CI Mode)
```bash
python run_all_kaggle.py --ultra-quick --seeds 42,123,456 --no-mlflow
```

### Full Benchmark
```bash
python run_all_kaggle.py --seeds 42,123,456,789,1011
```

### Resume from Checkpoint
```bash
python run_all_kaggle.py --resume
```

## Output Structure

Experiments save results to:
```
artifacts/
├── NN_SimpleMLP_MNIST_Adam_lr0.001_seed42.csv
├── NN_SimpleMLP_MNIST_Adam_lr0.001_seed42_meta.json
└── checkpoints/
    └── NN_SimpleMLP_MNIST_Adam_lr0.001_seed42_epoch10.pt
```

## Configuration

Experiments use JSON configs from `configs/`:
- `nn_tuning.json` - Neural network hyperparameter tuning
- `cifar10_tuning.json` - CIFAR-10 specific settings
- `benchmark_hyperparameters.json` - Benchmark default values

See `configs/README.md` for parameter documentation.
```

#### `src/utils/README.md` (1 hour)

```markdown
# Utilities Package

## Purpose
Shared utility functions and helpers used across experiments.

## Key Modules

### Configuration
- `config_validator.py` - JSON schema validation
- `experiment_config.py` - Typed configuration management
- `constants.py` - Project-wide constants

### File I/O
- `csv_utils.py` - Safe CSV reading/writing
- `file_safety.py` - Atomic file operations
- `filename.py` - Result filename parsing

### Reproducibility
- `reproducibility.py` - Seed setting, determinism checks

### Data Processing
- `transformed_subset.py` - Dataset transformation utilities
- `safe_len.py` - Safe length computation

### Analysis
- `convergence_detection.py` - Adaptive convergence criteria
- `metric_normalization.py` - Normalize metrics for comparison

### Fairness
- `fair_ablation.py` - Per-optimizer fair default LRs
- `fairness_check.py` - Validate experiment fairness

### Error Handling
- `error_handling_patterns.py` - Decorators for GPU safety
- `type_guards.py` - Runtime type checking

## Usage Examples

### Safe CSV Reading
```python
from src.utils.csv_utils import safe_read_csv
df = safe_read_csv("results/experiment.csv")
if df is not None:
    process(df)
```

### Configuration Validation
```python
from src.utils.config_validator import validate_config
config = load_json("config.json")
validate_config(config, "configs/config_schema.json")
```

### Convergence Detection
```python
from src.utils.convergence_detection import AdaptiveConvergenceDetector
detector = AdaptiveConvergenceDetector()
for epoch in range(epochs):
    result = detector.check_convergence(loss_history, grad_norms)
    if result.converged:
        break
```

## Testing

```bash
pytest tests/test_utils.py -v
```
```

#### `src/visualization/README.md` (1 hour)

Create similar structure for visualization package.

#### `src/analysis/README.md` (1 hour)

Create similar structure for analysis package.

#### `tests/README.md` (1 hour)

```markdown
# Test Suite

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_optimizers.py -v

# Quick smoke test (import safety)
python scripts/quick_validation_test.py --verbose

# Integration test (fast pipeline)
pytest tests/test_integration_quick_pipeline.py -v
```

## Test Categories

### Unit Tests
- `test_optimizers.py` - Optimizer correctness
- `test_pytorch_optimizers.py` - PyTorch wrapper tests
- `test_models.py` - Neural network architecture tests
- `test_utils.py` - Utility function tests

### Integration Tests
- `test_integration_quick_pipeline.py` - End-to-end pipeline
- `test_tuning_safety.py` - Hyperparameter tuning

### Safety Tests
- `test_import_safety.py` - No side effects on import
- `test_determinism.py` - Reproducibility checks

## Test Flags

- `--ultra-quick`: Minimal workload for CI
- `--no-mlflow`: Disable tracking for tests
```

#### `configs/README.md` (1 hour)

```markdown
# Configuration Guide

## Configuration Files

### `nn_tuning.json`
Neural network hyperparameter tuning configuration.

**Required Fields:**
- `dataset`: "MNIST" | "CIFAR10" | "IMDB"
- `model`: Model architecture name
- `seeds`: Array of random seeds (minimum 3)
- `batch_size`: Training batch size
- `sweeps`: Array of optimizer sweep configurations

**Example:**
```json
{
  "dataset": "MNIST",
  "model": "SimpleMLP",
  "seeds": [42, 123, 456],
  "batch_size": 128,
  "sweeps": [
    {
      "optimizer": "Adam",
      "lr_values": [0.0001, 0.001, 0.01],
      "weight_decay_values": [0.0, 0.0001, 0.001]
    }
  ]
}
```

### Parameter Ranges

#### Learning Rate (`lr_values`)
- **Range:** [1e-10, 10.0]
- **Per-Optimizer Defaults:**
  - SGD: 0.1
  - SGD Momentum: 0.01
  - Adam: 0.001
  - AdamW: 0.001

#### Weight Decay (`weight_decay_values`)
- **Range:** [0.0, 1.0]
- **Typical:** [0.0, 1e-5, 1e-4, 1e-3]

#### Momentum (`momentum_values`)
- **Range:** [0.0, 1.0]
- **Typical:** [0.0, 0.9, 0.95, 0.99]

### Validation

Validate configs before running:
```bash
python scripts/validate_configs.py --config configs/nn_tuning.json
python scripts/validate_config_schema.py
```

## Creating Custom Configs

1. Copy existing config as template
2. Modify parameters for your experiment
3. Validate using schema validator
4. Test with `--ultra-quick` mode first
```

---

### Task 1.3: Create Troubleshooting Guide (4 hours)

**File:** `docs/TROUBLESHOOTING.md`

```markdown
# Troubleshooting Guide

## Common Errors and Solutions

### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Causes:**
- Batch size too large
- Model too large
- Gradient accumulation without clearing

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python run_all_kaggle.py --batch-size 64  # Instead of 128
   ```

2. **Use gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Enable mixed precision training:**
   ```bash
   python run_all_kaggle.py --use-amp
   ```

4. **Monitor GPU memory:**
   ```bash
   watch -n 1 nvidia-smi
   ```

**Prevention:**
- Start with small batch size, increase gradually
- Use `--ultra-quick` mode for testing

---

### 2. Dataset Download Failures

**Error:**
```
ConnectionError: Failed to download dataset
```

**Solutions:**

1. **Use Kaggle datasets script:**
   ```bash
   python download_datasets_kaggle.py
   ```

2. **Manual download:**
   - MNIST: https://yann.lecun.com/exdb/mnist/
   - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
   - Place in `data/` directory

3. **Check proxy settings:**
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   ```

---

### 3. Configuration Validation Errors

**Error:**
```
ValueError: Invalid configuration: seeds must have at least 3 elements
```

**Solution:**
```bash
# Validate config
python scripts/validate_configs.py --config configs/nn_tuning.json

# Fix seeds
# Edit config to have ≥3 seeds: [42, 123, 456]
```

**Common Config Issues:**
- Seeds < 3: Add more seeds
- Learning rate out of range: Use [1e-10, 10.0]
- Missing required fields: Check against schema

---

### 4. Checkpoint Resume Failures

**Error:**
```
FileNotFoundError: Checkpoint not found
```

**Solutions:**

1. **Check checkpoint exists:**
   ```bash
   ls artifacts/checkpoints/
   ```

2. **Use correct resume behavior:**
   ```bash
   # Error if no checkpoint
   python run_all_kaggle.py --resume --resume-behavior error_if_no_checkpoint
   
   # Restart if no checkpoint
   python run_all_kaggle.py --resume --resume-behavior restart_if_no_checkpoint
   ```

3. **Verify checkpoint integrity:**
   ```python
   import torch
   checkpoint = torch.load("checkpoint.pt")
   print(checkpoint.keys())  # Should have 'model_state_dict', 'optimizer_state_dict', etc.
   ```

---

### 5. MLflow Tracking Issues

**Error:**
```
mlflow.exceptions.MlflowException: Run not found
```

**Solutions:**

1. **Disable MLflow for debugging:**
   ```bash
   python run_all_kaggle.py --no-mlflow
   ```

2. **Check tracking URI:**
   ```bash
   echo $MLFLOW_TRACKING_URI
   # Should be empty or point to valid directory
   ```

3. **View UI:**
   ```bash
   mlflow ui --backend-store-uri mlruns/
   # Open browser to http://localhost:5000
   ```

4. **Clear corrupted runs:**
   ```bash
   rm -rf mlruns/.trash/
   ```

---

### 6. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**

1. **Run from project root:**
   ```bash
   cd /path/to/GDSearch
   python run_all_kaggle.py
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   ```

3. **Check PYTHONPATH:**
   ```bash
   export PYTHONPATH=/path/to/GDSearch:$PYTHONPATH
   ```

---

### 7. NaN/Inf Loss During Training

**Error:**
```
WARNING: Non-finite gradients detected
```

**Causes:**
- Learning rate too high
- Numerical instability
- Exploding gradients

**Solutions:**

1. **Reduce learning rate:**
   ```bash
   python run_all_kaggle.py --lr 0.0001  # Instead of 0.01
   ```

2. **Enable gradient clipping:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Check input data normalization:**
   ```python
   # Data should be normalized to [0, 1] or standardized
   transform = transforms.Normalize(mean=[0.5], std=[0.5])
   ```

4. **Use mixed precision carefully:**
   - AMP can cause underflow with fp16
   - Try gradient scaling

---

### 8. Slow Training Performance

**Symptoms:**
- Training takes significantly longer than expected
- GPU utilization < 50%

**Solutions:**

1. **Check num_workers for data loading:**
   ```python
   train_loader = DataLoader(dataset, num_workers=4)  # Increase from 0
   ```

2. **Enable cudnn benchmarking:**
   ```python
   torch.backends.cudnn.benchmark = True
   ```

3. **Profile training loop:**
   ```bash
   python -m torch.utils.bottleneck run_all_kaggle.py --ultra-quick
   ```

4. **Check dataloader bottleneck:**
   ```bash
   # Monitor CPU usage during training
   htop
   ```

---

### 9. Test Accuracy Much Lower Than Expected

**Symptoms:**
- Training accuracy high (>90%)
- Test accuracy low (<70%)

**Causes:**
- Overfitting
- Train/test distribution mismatch
- Incorrect test-time augmentation

**Solutions:**

1. **Add regularization:**
   ```bash
   python run_all_kaggle.py --weight-decay 0.001
   ```

2. **Use dropout:**
   ```python
   model = SimpleMLP(dropout=0.5)
   ```

3. **Check data augmentation:**
   ```python
   # Training: augment
   # Testing: no augmentation
   ```

4. **Verify test set usage:**
   - Ensure no test data used during training
   - Check data splits are correct

---

## Getting Help

1. **Check existing issues:**
   - Search GitHub Issues: [link]

2. **Run validation scripts:**
   ```bash
   python scripts/quick_validation_test.py --verbose
   ```

3. **Enable debug logging:**
   ```bash
   python run_all_kaggle.py --log-level DEBUG
   ```

4. **Minimal reproducible example:**
   ```bash
   python run_all_kaggle.py --ultra-quick --seeds 42 --no-mlflow
   ```

5. **Collect diagnostics:**
   ```bash
   python -m torch.utils.collect_env
   nvidia-smi
   pip list | grep torch
   ```
```

---

### Task 1.4: Run Automated Checks (4 hours)

**Goal:** Establish baseline metrics and identify specific issues.

```bash
# 1. Run pydocstyle
pydocstyle src/ --convention=google > docs/pydocstyle_week1.txt
# Review output, categorize by severity

# 2. Run interrogate
interrogate -vv src/ > docs/interrogate_week1.txt
# Measure docstring coverage by module

# 3. Run mypy
mypy --strict src/ > docs/mypy_week1.txt
# Identify type hint gaps

# 4. Generate task list
python scripts/generate_doc_tasks.py
# Create file-by-file TODO list
```

---

### Task 1.5: Update MASTER_FIX_TRACKER.md (2 hours)

Add section documenting documentation improvements:

```markdown
## Documentation Improvements - Phase 1 (Week 1)

### Completed:
- [x] Complete docstrings for 13 optimizers in src/core/optimizers.py
- [x] Create 9 package README files
- [x] Create docs/TROUBLESHOOTING.md
- [x] Run baseline automated checks

### Metrics:
- Docstring coverage: 65% → 85%
- Package READMEs: 0/9 → 9/9
- Troubleshooting docs: 0 → 1 (100% coverage of common issues)

### Next Phase:
- Complete type hints in src/utils/
- Add Examples to PyTorch wrappers
- Create docs/ALGORITHMS.md
```

---

## Phase 2: High Priority (Week 2 - 40 hours)

### Task 2.1: Complete Type Hints (12 hours)

**Focus Files:**
1. `src/utils/filename.py` - Add complete type hints (2 hours)
2. `src/utils/plot_helpers.py` - Add complete type hints (2 hours)
3. `src/experiments/run_nn_experiment.py` - Complete Optional annotations (3 hours)
4. `run_all_kaggle.py` - Add type hints to main functions (5 hours)

**Pattern:**
```python
# Before
def process_data(data, config, device):
    pass

# After
def process_data(
    data: Union[np.ndarray, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Process data for training.
    
    Args:
        data: Input data as numpy array or PyTorch tensor
        config: Configuration dictionary with keys:
            - 'normalize': bool, whether to normalize
            - 'dtype': torch.dtype, target data type
        device: Target device for computation
        
    Returns:
        Processed tensor on specified device
    """
    pass
```

**Validation:**
```bash
mypy --strict src/utils/filename.py
mypy --strict src/utils/plot_helpers.py
# Should pass with zero errors
```

---

### Task 2.2: Add Examples to PyTorch Wrappers (10 hours)

**File:** `src/core/pytorch_optimizers.py`

For each wrapper class, add comprehensive Examples section:

```python
class AdamWrapper(Optimizer):
    """
    PyTorch wrapper for custom Adam optimizer.
    
    [... existing docstring ...]
    
    Example:
        >>> import torch
        >>> from src.core.pytorch_optimizers import AdamWrapper
        >>> 
        >>> # Basic usage
        >>> model = torch.nn.Linear(10, 1)
        >>> optimizer = AdamWrapper(model.parameters(), lr=0.001)
        >>> 
        >>> # Training loop
        >>> for epoch in range(epochs):
        >>>     optimizer.zero_grad()
        >>>     output = model(data)
        >>>     loss = criterion(output, target)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>> 
        >>> # With learning rate scheduler
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        >>> for epoch in range(epochs):
        >>>     train_epoch(model, optimizer)
        >>>     scheduler.step()
        >>> 
        >>> # Checkpoint saving
        >>> torch.save({
        >>>     'model_state_dict': model.state_dict(),
        >>>     'optimizer_state_dict': optimizer.state_dict(),
        >>>     'epoch': epoch,
        >>>     'loss': loss,
        >>> }, 'checkpoint.pt')
    """
```

**Wrappers to document:**
1. SGDWrapper
2. SGDMomentumWrapper
3. AdamWrapper
4. AdamWWrapper
5. RMSPropWrapper
6. SAMWrapper (emphasize closure requirement)
7. LookaheadWrapper
8. AdaBoundWrapper
9. RAdamWrapper
10. LAMBWrapper

---

### Task 2.3: Create docs/ALGORITHMS.md (8 hours)

**File:** `docs/ALGORITHMS.md`

```markdown
# Optimizer Algorithm Reference

## Overview
Comprehensive reference for all optimization algorithms implemented in GDSearch.

## Table of Contents
- [SGD Family](#sgd-family)
- [Adaptive Methods](#adaptive-methods)
- [Advanced Optimizers](#advanced-optimizers)
- [Convergence Rates](#convergence-rates)
- [Hyperparameter Recommendations](#hyperparameter-recommendations)

---

## SGD Family

### Stochastic Gradient Descent (SGD)

**Algorithm:**
```
θ_{t+1} = θ_t - η · ∇L(θ_t)
```

**Key Properties:**
- Convergence rate: O(1/k) for strongly convex
- No momentum or adaptivity
- Simple but requires careful LR tuning

**Reference:**
Robbins, H., & Monro, S. (1951). A stochastic approximation method. The annals of mathematical statistics, 400-407.

**When to Use:**
- Simple convex problems
- When you have time to tune learning rate
- Baseline comparison

**Hyperparameters:**
- Learning rate: [0.001, 0.1] typical range
- Weight decay: [0.0, 0.01] for regularization

---

### SGD with Momentum

**Algorithm:**
```
v_{t+1} = β · v_t + ∇L(θ_t)
θ_{t+1} = θ_t - η · v_{t+1}
```

**Key Properties:**
- Accelerates convergence: O(1/√κ) for κ-conditioned quadratics
- Reduces oscillations
- Momentum coefficient β typically 0.9

**Reference:**
Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5), 1-17.

**When to Use:**
- Ill-conditioned problems
- Want faster convergence than vanilla SGD
- Standard choice for deep learning

---

[Continue for all 13 optimizers...]

---

## Convergence Rates Summary

| Optimizer | Strongly Convex | General Convex | Non-Convex |
|-----------|-----------------|----------------|------------|
| SGD | O(1/k) | O(1/√k) | O(1/√k) to stationary |
| Momentum | O(1/√κ) | O(1/k) | - |
| Adam | O(1/√k) | O(1/√k) | O(1/√k) to stationary |
| SAM | Empirical improvement | - | Flatter minima |

---

## Hyperparameter Recommendations

### Learning Rate Selection

| Optimizer | MNIST | CIFAR-10 | ImageNet | NLP |
|-----------|-------|----------|----------|-----|
| SGD | 0.1 | 0.1 | 0.1 | 1.0 |
| Momentum | 0.01 | 0.1 | 0.1 | 1.0 |
| Adam | 0.001 | 0.001 | 0.001 | 0.0001 |
| AdamW | 0.001 | 0.001 | 0.0001 | 0.0001 |

### Weight Decay

- **L2 Regularization (SGD, Adam):** [1e-5, 1e-3]
- **Decoupled (AdamW):** [0.01, 0.1]

### Momentum/Beta Values

- **β1 (first moment):** 0.9 (standard), [0.8, 0.95] for tuning
- **β2 (second moment):** 0.999 (standard), [0.99, 0.9999] for tuning

---

## Computational Costs

| Optimizer | Memory Overhead | Time per Step | Notes |
|-----------|-----------------|---------------|-------|
| SGD | 1x | 1x | Baseline |
| Momentum | 1.5x | 1x | Velocity buffer |
| Adam | 2x | 1.2x | Two moment buffers |
| SAM | 2x | 2x | Double backward pass |
| Lookahead | 2x | 1x (amortized) | Slow weights |

---

## References

[Complete bibliography of all papers]
```

---

### Task 2.4: Create configs/README.md (6 hours)

Already outlined in Task 1.2, but expand with more examples and parameter explanations.

---

### Task 2.5: Run Phase 2 Validation (4 hours)

```bash
# Check type hint coverage improved
mypy --strict src/ > docs/mypy_week2.txt
diff docs/mypy_week1.txt docs/mypy_week2.txt

# Check docstring coverage
interrogate -vv src/ > docs/interrogate_week2.txt

# Verify Examples compile
python scripts/validate_docstring_examples.py

# Update metrics in MASTER_FIX_TRACKER.md
```

---

## Phase 3: Polish (Week 3 - 40 hours)

### Task 3.1: Add Inline Comments to Complex Algorithms (12 hours)

**Focus Areas:**
1. `src/core/optimizers.py` - Mathematical operations
2. `src/experiments/training_loops.py` - Training logic
3. `src/analysis/hessian_analysis.py` - Eigenvalue computation
4. `run_all_kaggle.py` - Experiment orchestration

**Pattern:**
```python
# BAD - Describes what code does
x = x * 0.9 + 0.1 * y  # Multiply x by 0.9 and add 0.1 times y

# GOOD - Explains why and references theory
# Exponential moving average with decay rate 0.9 (Kingma & Ba 2015, Eq. 3)
# Used for bias-corrected first moment estimate in Adam
x = x * 0.9 + 0.1 * y
```

---

### Task 3.2: Create docs/API_REFERENCE.md (10 hours)

Generate comprehensive API documentation:

```bash
# Install sphinx
pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme

# Generate API docs
sphinx-apidoc -o docs/api src/
cd docs/
make html

# Create API_REFERENCE.md with links
```

---

### Task 3.3: Run pydocstyle and Fix All Errors (10 hours)

```bash
# Get error count
pydocstyle src/ --convention=google | wc -l

# Fix by priority:
# 1. Missing docstrings (D100, D101, D102, D103)
# 2. Incomplete docstrings (D200, D205, D400)
# 3. Formatting issues (D212, D213)

# Verify zero errors
pydocstyle src/ --convention=google
# Should output: Success! No issues found.
```

---

### Task 3.4: Achieve 100% Docstring Coverage (8 hours)

```bash
# Measure current coverage
interrogate -vv src/ --fail-under=90

# Generate missing docstring report
interrogate -vv src/ --generate-badge docs/docstring_coverage.svg

# Fix remaining gaps
# Target: ≥95% coverage
```

---

## Validation Checklist

Before declaring Phase 3 complete:

- [ ] `pydocstyle src/ --convention=google` returns 0 errors
- [ ] `interrogate src/ --fail-under=95` passes
- [ ] `mypy --strict src/` has <50 errors (down from ~200)
- [ ] All 9 package README files exist and are complete
- [ ] `docs/TROUBLESHOOTING.md` covers top 10 errors
- [ ] `docs/ALGORITHMS.md` has references for all 13 optimizers
- [ ] `docs/API_REFERENCE.md` is auto-generated and up-to-date
- [ ] All docstrings have Examples sections (80%+ compliance)

---

## Metrics Tracking

Create `docs/documentation_metrics.csv` to track progress:

```csv
Week,Docstring_Coverage_%,Type_Hint_Coverage_%,Package_READMEs,pydocstyle_Errors,mypy_Errors
Baseline,65,60,0,450,200
Week1,85,65,9,180,190
Week2,92,85,9,50,80
Week3,98,92,9,0,45
```

---

## Deliverables Summary

### Phase 1 (Week 1)
- ✅ Complete docstrings for 13 optimizers
- ✅ 9 package README files
- ✅ `docs/TROUBLESHOOTING.md`
- ✅ Baseline metrics established

### Phase 2 (Week 2)
- ✅ Complete type hints in utils
- ✅ Examples in all PyTorch wrappers
- ✅ `docs/ALGORITHMS.md`
- ✅ `configs/README.md`

### Phase 3 (Week 3)
- ✅ Inline comments in complex code
- ✅ `docs/API_REFERENCE.md`
- ✅ Zero pydocstyle errors
- ✅ ≥95% docstring coverage

---

## Success Criteria

**Documentation Quality Score: ≥90/100**

| Metric | Target | Weight |
|--------|--------|--------|
| Docstring Coverage | ≥95% | 30% |
| Type Hint Coverage | ≥90% | 20% |
| Package READMEs | 9/9 | 15% |
| pydocstyle Errors | 0 | 15% |
| Algorithm References | 13/13 | 10% |
| Troubleshooting Coverage | 100% | 10% |

**Final Validation:**
- [ ] External reviewer can understand codebase from docs alone
- [ ] New contributors can add optimizer without asking questions
- [ ] All experiments reproducible from documentation
- [ ] Publication-ready algorithm explanations

---

## Maintenance Plan

**After completion:**

1. **Pre-commit hook:**
   ```bash
   # .git/hooks/pre-commit
   pydocstyle src/ --convention=google
   ```

2. **CI integration:**
   ```yaml
   # .github/workflows/docs.yml
   - name: Check documentation
     run: |
       pydocstyle src/ --convention=google
       interrogate src/ --fail-under=95
   ```

3. **Monthly review:**
   - Run interrogate and update badge
   - Check for new undocumented code
   - Update ALGORITHMS.md with new papers

---

**End of Roadmap**  
**Estimated Total Time:** 120 hours over 3 weeks  
**Expected Outcome:** Publication-ready documentation with ≥90% quality score
