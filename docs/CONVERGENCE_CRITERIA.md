# Convergence Detection Criteria

## Overview

GDSearch uses adaptive convergence detection with problem-specific thresholds optimized for different optimization landscapes. This document explains the rationale and default values.

## Problem-Specific Defaults

### 2D Test Functions

**Typical Use:** Rosenbrock, Rastrigin, Ackley, Sphere, etc.

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Absolute Loss** | `1e-6` | Test functions have near-zero global optima |
| **Gradient Norm** | `1e-6` | Smooth, deterministic landscapes enable tight convergence |
| **Relative Tolerance** | `1%` | Can achieve precise convergence without overfitting concerns |
| **Plateau Patience** | `1e-8` | Detect plateaus at very fine resolution |

**Code Location:** [src/utils/convergence_detection.py](../src/utils/convergence_detection.py)

```python
if problem_type == 'test_function':
    detector = AdaptiveConvergenceDetector(
        absolute_loss_threshold=1e-6,
        gradient_threshold=1e-6,
        relative_tolerance=0.01,  # 1%
        plateau_tolerance=1e-8
    )
```

**Why These Values:**
- Test functions are **deterministic** (no stochastic gradients)
- Gradients are **smooth** and **analytically computable**
- Global optima are **known** and typically zero
- Convergence can be **mathematically proven** (e.g., Rosenbrock at (1,1) = 0)

---

### Neural Network Training

**Typical Use:** MNIST, CIFAR-10, NLP, Medical Segmentation

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Absolute Loss** | `1e-3` | Rarely reach near-zero loss due to regularization |
| **Gradient Norm** | `1e-4` | Stochastic gradients are noisy |
| **Relative Tolerance** | `5%` | Loose convergence prevents overfitting |
| **Plateau Patience** | `1e-8` | Same as 2D (loss scale normalized) |

**Code Location:** [src/utils/convergence_detection.py](../src/utils/convergence_detection.py)

```python
if problem_type == 'neural_network':
    detector = AdaptiveConvergenceDetector(
        absolute_loss_threshold=1e-3,
        gradient_threshold=1e-4,
        relative_tolerance=0.05,  # 5%
        plateau_tolerance=1e-8
    )
```

**Why These Values:**
- **Non-zero optima:** Regularization (weight decay, label smoothing) prevents zero loss
- **Stochastic gradients:** Mini-batch SGD introduces variance
- **Generalization:** Tight convergence may cause overfitting
- **Early stopping:** Validation plateau detection is more important than training convergence

---

## Criterion Definitions

### 1. Absolute Loss Threshold

**Definition:** Stop when `loss < threshold`

**Formula:**
```python
if loss < absolute_loss_threshold:
    return True  # Converged
```

**Use Cases:**
- Test functions with known zero optima
- Classification with perfect separability (rare in practice)

**Limitations:**
- Not scale-invariant (depends on loss magnitude)
- May never trigger for neural networks with regularization

---

### 2. Gradient Norm Threshold

**Definition:** Stop when `||∇L|| < threshold`

**Formula:**
```python
grad_norm = sqrt(sum(g_i^2 for all parameters))
if grad_norm < gradient_threshold:
    return True  # Converged to critical point
```

**Use Cases:**
- Detecting local minima, saddle points, or global minima
- Verifying first-order optimality conditions

**Limitations:**
- **Saddle points** have zero gradient but are not minima
- **Noisy gradients** (SGD) may never reach threshold
- **Gradient clipping** artificially reduces norm

---

### 3. Relative Tolerance

**Definition:** Stop when loss is within `relative_tolerance` of best seen loss

**Formula:**
```python
relative_improvement = (best_loss - current_loss) / abs(best_loss)
if relative_improvement < relative_tolerance:
    return True  # No significant improvement
```

**Use Cases:**
- Neural networks where absolute loss is unknown
- Comparing convergence across different datasets
- Early stopping based on validation loss

**Example:**
```python
# 5% relative tolerance
best_loss = 0.200
current_loss = 0.201
relative_improvement = (0.200 - 0.201) / 0.200 = -0.005 = -0.5%
# Not converged (worse than best)

current_loss = 0.199
relative_improvement = (0.200 - 0.199) / 0.200 = 0.005 = 0.5%
# Converged (within 5% of best)
```

---

### 4. Plateau Detection

**Definition:** Stop when loss changes by less than `plateau_tolerance` over `patience` steps

**Formula:**
```python
loss_window = [loss_t, loss_{t-1}, ..., loss_{t-patience}]
loss_variance = var(loss_window)
if loss_variance < plateau_tolerance:
    return True  # Loss has plateaued
```

**Use Cases:**
- Detecting when optimizer has stalled
- Early stopping for neural networks
- Identifying when to increase learning rate (learning rate scheduling)

**Parameters:**
- `patience`: Number of steps to check (default: 10 epochs)
- `plateau_tolerance`: Variance threshold (default: 1e-8)

---

## Configuration

### Per-Experiment Configuration

Override defaults in config files:

```json
{
  "convergence": {
    "absolute_loss_threshold": 1e-4,
    "gradient_threshold": 1e-5,
    "relative_tolerance": 0.02,
    "plateau_patience": 20,
    "plateau_tolerance": 1e-7
  }
}
```

### CLI Override

```bash
# Use tighter convergence for test function
python run_all_kaggle.py --experiments 2d \
    --convergence-threshold 1e-8 \
    --gradient-threshold 1e-8

# Use looser convergence for neural network
python run_all_kaggle.py --experiments mnist \
    --convergence-threshold 1e-2 \
    --gradient-threshold 1e-3
```

---

## Rationale: Why Different Thresholds?

### Problem Landscape Differences

| Property | 2D Test Functions | Neural Networks |
|----------|-------------------|-----------------|
| **Determinism** | Fully deterministic | Stochastic (mini-batch sampling) |
| **Gradient Smoothness** | Smooth (analytical) | Noisy (approximated) |
| **Known Optima** | Usually zero | Unknown (depends on regularization) |
| **Overfitting Risk** | None (no generalization) | High (validation loss diverges) |
| **Convergence Proof** | Theoretically provable | No global convergence guarantee |

### Example: Rosenbrock vs MNIST

**Rosenbrock (2D Test Function):**
```python
# Global minimum: f(1, 1) = 0
# Gradient: ∇f = 0 at optimum
# Convergence: Can achieve loss < 1e-10 in 1000 iterations
detector = AdaptiveConvergenceDetector(absolute_loss_threshold=1e-6)
```

**MNIST (Neural Network):**
```python
# "Optimal" loss: ~0.03 (with label smoothing, weight decay)
# Gradient: Never exactly zero (stochastic, mini-batch variance)
# Convergence: Loss plateaus around 0.05, overfits if trained longer
detector = AdaptiveConvergenceDetector(
    absolute_loss_threshold=1e-3,  # Will never trigger
    relative_tolerance=0.05,        # Triggers when within 5% of best
    gradient_threshold=1e-4         # Stochastic gradients never reach 1e-6
)
```

---

## Scientific Validity

### Why Not Use Same Thresholds Everywhere?

**Attempted in Early Development:**
- Used `1e-6` for all experiments
- **Result:** Neural networks **never converged** (ran for maximum epochs)
- **Reason:** Stochastic gradients + regularization prevent reaching 1e-6 loss

**Lesson Learned:**
> Convergence criteria must match problem characteristics. Using 2D thresholds for neural networks results in training that never stops.

### Comparison Fairness

**Q:** Does using different thresholds make optimizer comparisons unfair?

**A:** No, because:
1. **Same threshold within problem type:** All MNIST experiments use `1e-3`
2. **Relative metrics:** We compare convergence **speed** (epochs to plateau), not absolute loss
3. **Fair defaults:** Learning rates are tuned per-optimizer using same methodology

**Invalid Comparison:**
```python
# ❌ WRONG: Comparing 2D Rosenbrock vs NN MNIST
rosenbrock_iterations = 1000  # Converged to 1e-6
mnist_iterations = never      # Never reached 1e-6
# Conclusion: "MNIST training failed" — INCORRECT!
```

**Valid Comparison:**
```python
# ✅ CORRECT: Comparing within problem type
sgd_epochs_to_95pct = 50   # 95% of best loss
adam_epochs_to_95pct = 30  # 95% of best loss
# Conclusion: "Adam converges 1.67x faster than SGD on MNIST"
```

---

## Debugging Convergence Issues

### Issue: "Training never converges"

**Symptoms:**
- Runs for maximum epochs (e.g., 100) without stopping
- Loss decreases slowly but never triggers convergence

**Diagnosis:**
```python
# Check convergence settings
python -c "from src.utils.convergence_detection import AdaptiveConvergenceDetector; print(AdaptiveConvergenceDetector().__dict__)"

# Enable verbose logging
python run_all_kaggle.py --experiments mnist --verbose
```

**Solutions:**
1. **Loosen threshold:** Increase `absolute_loss_threshold` from `1e-3` to `1e-2`
2. **Use relative tolerance:** Switch to `relative_tolerance=0.05` instead of absolute
3. **Reduce patience:** Decrease `plateau_patience` from 20 to 10 epochs

---

### Issue: "Converges too early"

**Symptoms:**
- Training stops after 5-10 epochs
- Validation accuracy still improving when training stops

**Diagnosis:**
```python
# Check if plateau detected prematurely
grep "Converged" artifacts/log.txt
```

**Solutions:**
1. **Tighten threshold:** Decrease `relative_tolerance` from `0.05` to `0.01`
2. **Increase patience:** Increase `plateau_patience` from 10 to 20 epochs
3. **Use validation loss:** Ensure convergence checks validation, not training loss

---

## Related Documentation

- [src/utils/convergence_detection.py](../src/utils/convergence_detection.py): Implementation
- [docs/guides/EXPERIMENT_EXECUTION_GUIDE.md](guides/EXPERIMENT_EXECUTION_GUIDE.md): Running experiments
- [configs/config_schema.json](../configs/config_schema.json): Configuration schema

---

**Last Updated:** February 2, 2026  
**Maintainer:** GDSearch Team
