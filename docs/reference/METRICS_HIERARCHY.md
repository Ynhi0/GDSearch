# Metrics Hierarchy for Convergence Analysis

**Senior Principal Software Engineer — Measurement Validity Framework**  
**Date:** January 6, 2026  
**Purpose:** Define valid metrics for each research question and prevent common logical fallacies

---

## The Fundamental Logical Flaw

**Common Mistake:** Claiming "Optimizer A has better convergence rate than Optimizer B" while showing graphs of **Test Accuracy**.

**Why This is Wrong:**
- **Convergence Rate** (optimization theory) = How fast **TRAINING LOSS** → minimum
- **Generalization** (learning theory) = Gap between **TRAIN** and **TEST** performance

These are **INDEPENDENT** properties:
- An algorithm can have **slow convergence** but **good generalization** (e.g., SGD+Momentum with small batch)
- An algorithm can have **fast convergence** but **poor generalization** (e.g., Adam with large batch, overfitting)

---

## Correct Metrics for Each Research Question

### Research Question 1: "What is the convergence rate?" (Tốc độ hội tụ)

**Primary Metric:** Training Loss vs. Iterations  
**Secondary Metric:** Gradient Norm (for non-convex problems)  
**Irrelevant Metric:** Test Accuracy ❌ (measures generalization, NOT convergence speed)

**Valid Comparison:**
```python
# Correct: Compare training loss curves
if adam_train_loss_epoch_25 < sgd_train_loss_epoch_25:
    print("Adam converges faster than SGD (reaches lower training loss in fewer epochs)")
```

**Invalid Comparison:**
```python
# WRONG: Using test accuracy to claim convergence speed
if adam_test_acc > sgd_test_acc:
    print("Adam has better convergence")  # ❌ This measures GENERALIZATION, not convergence
```

**Example Correct Statement:**
> "SGD with momentum converges to training loss < 0.01 in 25 epochs, while vanilla SGD requires 40 epochs (60% speedup)."

**Example INCORRECT Statement:**
> "SGD with momentum has better convergence because its test accuracy is 2% higher."  
> *(Higher test accuracy means better generalization OR better hyperparameter tuning, NOT faster convergence.)*

---

### Research Question 2: "Which optimizer generalizes better?" (Khả năng tổng quát hóa)

**Primary Metric:** Generalization Gap = Test Loss - Train Loss  
**Secondary Metric:** Test Accuracy (at SAME training loss level)  
**Control Variable:** Ensure all optimizers reach same training loss before comparing  

**Valid Comparison (Method 1 - Fixed Training Loss):**
```python
# Compare test accuracy when all optimizers reach train_loss = 0.05
adam_test_acc_at_train_0_05 = 89.2%
sgd_test_acc_at_train_0_05 = 91.1%
# Conclusion: SGD generalizes better (finds flatter minima)
```

**Valid Comparison (Method 2 - Generalization Gap):**
```python
# At epoch 50 (fixed compute budget)
adam_gen_gap = adam_test_loss - adam_train_loss = 0.12
sgd_gen_gap = sgd_test_loss - sgd_train_loss = 0.08
# Conclusion: SGD has smaller generalization gap (2x better)
```

**Example Correct Statement:**
> "When both optimizers reach train_loss=0.05, Adam has gen_gap=0.12 while Momentum has gen_gap=0.08, suggesting Momentum finds flatter minima (Keskar et al. 2017)."

---

### Research Question 3: "Which optimizer is best for practitioners?" (Hiệu suất thực tế)

**Primary Metric:** Test Accuracy at fixed compute budget (e.g., 50 epochs)  
**Justification:** This is the metric practitioners care about (but it conflates convergence + generalization + hyperparameter tuning quality)

**Valid Comparison:**
```python
# After 50 epochs with best hyperparameters found via grid search
adam_final_test_acc = 92.3%
sgd_momentum_final_test_acc = 91.8%
# Conclusion: Adam wins for this specific task/budget/tuning (but we don't know WHY)
```

**Required Disclaimer:**
> "Test accuracy comparisons reflect the combined effect of convergence speed, generalization ability, and hyperparameter sensitivity. To isolate convergence rate, see Figure X (training loss curves)."

---

## Metric Validity by Problem Type

| Metric | 2D Deterministic | Neural Network (Stochastic) | Validity |
|--------|------------------|----------------------------|----------|
| **Distance to Optimum** ||x_t - x*|| | ✅ Valid (x* known) | ❌ Invalid (x* unknown) | 2D only |
| **Loss Regret** f(x_t) - f(x*) | ✅ Valid | ❌ Invalid | 2D only |
| **Gradient Norm** ||∇f|| | ✅ Valid (→ 0) | ⚠️ Valid but never → 0 (noise floor) | Different interpretation |
| **Training Loss** f(x_t) | ✅ Valid | ✅ Valid | Universal |
| **Test Accuracy** | ❌ N/A | ✅ Valid | NN only |
| **Generalization Gap** | ❌ N/A | ✅ Valid | NN only |
| **Hessian Eigenvalues** λ_min, λ_max | ✅ Exact | ⚠️ Approximate (Lanczos) | Different precision |

---

## The "Gradient Norm" Trap (Critical Nuance)

### For 2D Deterministic Functions:
```python
# Correct convergence criterion
if gradient_norm < 1e-6:
    print("Converged to stationary point (∇f ≈ 0)")
```

### For Neural Networks (Stochastic):
```python
# WRONG: Gradient norm NEVER goes to zero due to mini-batch noise
if gradient_norm < 1e-6:
    print("Converged")  # ❌ This will never trigger or will trigger prematurely

# CORRECT: Use loss plateau or gradient noise stabilization
if abs(loss_epoch_t - loss_epoch_t_minus_5) < 1e-4:
    print("Training loss has plateaued (practical convergence)")
```

**Theoretical Background:**  
For mini-batch SGD, the gradient norm reaches a **noise floor**:
```
E[||∇f_mini-batch||²] ≈ ||∇f_true||² + σ²/batch_size
```
Even at a minimum (||∇f_true|| = 0), the measured gradient norm is √(σ²/batch_size), not zero.

**Codebase Implementation:**  
File: `src/analysis/gradient_noise_analysis.py`
```python
def estimate_noise_floor(grad_history, window=100):
    """Estimate the gradient noise floor (minimum achievable ||∇f||)."""
    return np.std(grad_history[-window:])  # Standard deviation of late-stage gradients
```

---

## Epoch vs. Iteration Scaling (Critical Distinction)

### The Math:
For CIFAR10 (50,000 images) with batch size 128:
```
1 Epoch = 50,000 / 128 = 391 Steps (Iterations)
```

### Implications for Convergence Rate Analysis:

**Theoretical Bounds Use STEPS (k):**
```
SGD: E[f(x_k) - f(x*)] ≤ O(1/k)  # k = iteration count
```

**Deep Learning Papers Often Report EPOCHS:**
```
"ResNet-18 reaches 92% accuracy after 90 epochs"  # = 35,190 steps for CIFAR10
```

### Correct Usage:

**For Theory Validation Plots (X-axis = Steps):**
```python
# Correct: Fit O(1/k) curve to step-wise loss
plt.plot(iterations, train_loss, label='Measured')
plt.plot(iterations, C/iterations, label='O(1/k) theory', linestyle='--')
plt.xlabel('Iterations (k)')
```

**For Practical Comparison Plots (X-axis = Epochs):**
```python
# Correct: Compare optimizers at epoch-level granularity
plt.plot(epochs, test_accuracy, label='Adam')
plt.xlabel('Epochs')
```

**⚠️ NEVER Mix Units:**
```python
# WRONG: Fitting a curve meant for steps onto an epoch-scale axis
plt.plot(epochs, train_loss)
plt.plot(epochs, C/epochs, label='O(1/k)')  # ❌ This is mathematically incorrect
# Should be: C / (epochs * steps_per_epoch)
```

**Codebase Implementation:**  
File: `src/experiments/theory_practice_validation.py` (Line 198)
```python
# Correctly extracts iteration count for theory comparison
iteration_count = len(loss_history)  # Total steps, not epochs
```

---

## Wall-Clock Time vs. Iteration Trade-off

### The Hidden Computational Cost:

**Adam Computation (per step):**
1. Compute gradient ∇f
2. Update first moment: m_t = β_1 * m_{t-1} + (1-β_1) * ∇f
3. Update second moment: v_t = β_2 * v_{t-1} + (1-β_2) * ∇f²
4. Bias correction: m̂ = m_t / (1-β_1^t), v̂ = v_t / (1-β_2^t)
5. Parameter update: θ_t = θ_{t-1} - α * m̂ / (√v̂ + ε)

**SGD Computation (per step):**
1. Compute gradient ∇f
2. Parameter update: θ_t = θ_{t-1} - α * ∇f

**Cost Ratio:** Adam requires **~3x more operations** per step (2 momentum updates + sqrt + division).

### Valid Comparisons:

**For Theoretical Analysis (Steps):**
```python
# Compare iteration complexity (independent of hardware)
adam_steps_to_loss_0_01 = 5000
sgd_steps_to_loss_0_01 = 8000
print(f"Adam requires 37.5% fewer steps")  # Valid theoretical comparison
```

**For Practical Recommendations (Wall-Clock Time):**
```python
# Compare actual training time (hardware-dependent)
adam_time_to_loss_0_01 = 120 seconds
sgd_time_to_loss_0_01 = 100 seconds
print(f"SGD is 20% faster in wall-clock time")  # Valid practical comparison
```

**Recommendation:** Show BOTH in your thesis:
- **Figure 4.1:** Training Loss vs. Iterations (theory comparison)
- **Figure 4.2:** Training Loss vs. Wall-Clock Time (practical comparison)

**Codebase Implementation:**  
File: `scripts/compute_tradeoffs.py`
```python
def analyze_time_vs_steps(results_df):
    """Compare iteration efficiency vs. wall-clock efficiency."""
    # Plots both metrics side-by-side
```

---

## Distance to Optimum: Valid Use Cases

### ✅ Valid (2D Functions):
```python
# Rosenbrock: x* = (1, 1) is known
distance_to_opt = np.linalg.norm(x_current - x_star)
plt.plot(iterations, distance_to_opt)
plt.ylabel('||x_t - x*||')
plt.title('Distance to Known Optimum')
```

### ❌ Invalid (Neural Networks):
```python
# ResNet-18: x* is UNKNOWN (11M-dimensional non-convex landscape)
distance_to_opt = np.linalg.norm(params_current - params_optimal)  # ❌ What is params_optimal?
# This metric CANNOT be computed for neural networks
```

**Codebase Implementation:**  
File: `src/visualization/create_separate_plots.py`
```python
# Line ~120: Distance to optimum plot
if test_function in ['rosenbrock', 'sphere', 'quadratic']:
    plot_distance_to_optimum(...)  # Only for 2D functions with known x*
else:
    # Skip this plot for neural networks
    pass
```

**Thesis Presentation Rule:**
- **Chapter 3 (2D Experiments):** Include "Distance to Optimum" plot
- **Chapter 4 (Neural Networks):** Use "Training Loss" instead (never mention "distance to optimum")

---

## Implications for Thesis Structure

### ✅ CORRECT Thesis Organization:

**Chapter 3: Convergence Rate Analysis (Optimization Theory)**  
- **Metric:** Training Loss vs. Iterations (Steps)  
- **Figures:** Training loss curves (NOT test accuracy)  
- **Theory Comparison:** O(1/k) vs O(1/√κ) fits on TRAINING LOSS  
- **Scope:** 2D functions + Neural Networks (separate sections)

**Chapter 4: Generalization Analysis (Learning Theory)**  
- **Metric:** Generalization Gap = Test Loss - Train Loss  
- **Figures:** Generalization gap curves, test accuracy (at fixed train loss)  
- **Theory Comparison:** Sharpness metrics, flatness analysis  
- **Scope:** Neural Networks only (2D functions have no train/test split)

**Chapter 5: Practical Recommendations**  
- **Metric:** Test Accuracy at Fixed Compute Budget (50 epochs)  
- **Figures:** Final test accuracy bar charts, wall-clock time comparisons  
- **Justification:** "This metric conflates convergence + generalization + hyperparameter tuning, but reflects end-to-end practical performance."  
- **Scope:** Neural Networks only

---

## Defense Preparation: Anticipated Questions

### Q1: "Why is Adam's test accuracy higher if SGD has better convergence rate?"

**A:** "Convergence rate measures training loss reduction speed (optimization), while test accuracy measures generalization (learning theory). Adam converges faster (reaches train_loss=0.01 in 60% fewer steps, Figure 3.2) but generalizes worse (generalization gap 0.12 vs 0.08, Figure 4.1). The higher test accuracy reflects our hyperparameter tuning finding better learning rates for Adam on this specific task."

---

### Q2: "You claim momentum has O(1/√κ) convergence, but your Figure X shows epochs, not iterations."

**A:** "Thank you for catching that. Figure X is intended for practical comparison (epochs), while Figure Y validates the O(1/√κ) theory using iteration count (1 epoch = 391 steps for CIFAR10). I should clarify the axis units in the caption."

---

### Q3: "Can you plot the distance to the optimum for ResNet-18?"

**A:** "Unfortunately, no. For 2D Rosenbrock, the global optimum (1,1) is known analytically, so we can compute ||x_t - x*|| exactly (Figure 3.3). For ResNet-18, the 11-million-dimensional loss landscape is non-convex with unknown global minimum, making this metric mathematically undefined. Instead, we use training loss (Figure 4.1) as a proxy for optimization progress."

---

## Summary: Metrics Decision Tree

```
Research Question?
│
├─ "Which optimizer converges FASTER?"
│  └─ Metric: Training Loss vs. Iterations
│     Figures: Loss curves (NOT test accuracy)
│
├─ "Which optimizer GENERALIZES better?"
│  └─ Metric: Generalization Gap (Test - Train)
│     Figures: Gap curves, test accuracy at SAME train loss
│
├─ "Which optimizer is BEST overall?"
│  └─ Metric: Test Accuracy at Fixed Compute Budget
│     Figures: Final accuracy bar charts
│
└─ "Does the implementation match THEORY?"
   └─ Metric: Depends on problem class
      ├─ 2D Convex: Distance to optimum, gradient norm
      ├─ 2D Non-Convex: Gradient norm (→ 0)
      └─ Neural Network: Training loss plateau (gradient norm has noise floor)
```

---

## Code Audit: Correct Metric Usage

### ✅ Correctly Implemented:

**File:** `src/experiments/theory_practice_validation.py` (Line 207)
```python
if 'train_loss' in df.columns:
    loss_history = df['train_loss'].values  # ✅ Uses training loss for convergence
```

**File:** `src/experiments/weight_decay_ablation.py` (Line 213)
```python
gen_gaps.append(float(test_loss - train_loss))  # ✅ Correct generalization gap
```

**File:** `src/analysis/gradient_noise_analysis.py`
```python
def estimate_noise_floor(grad_history, window=100):
    return np.std(grad_history[-window:])  # ✅ Acknowledges gradient noise in SGD
```

### ⚠️ Requires Clarification:

**File:** `src/visualization/create_separate_plots.py` (Line ~120)
```python
# Needs explicit check to prevent distance_to_optimum plot for neural networks
if task_type == '2d_function' and optimum_known:
    plot_distance_to_optimum(...)
# Should NOT be called for ResNet-18
```

---

## Conclusion

The hierarchy is:
1. **Convergence** (optimization speed) → Training Loss
2. **Generalization** (learning quality) → Test Loss - Train Loss
3. **Practical Performance** (end result) → Test Accuracy

Conflating these three metrics is the #1 mistake in optimizer comparison papers. This codebase correctly separates them—now the thesis documentation must reflect this rigor.
