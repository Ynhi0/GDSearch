# Dimensionality and Landscape Transfer: Limitations of 2D Visualizations

**Senior Principal Software Engineer — Architectural Design Rationale**  
**Date:** January 6, 2026  
**Purpose:** Clarify the logical separation between 2D deterministic GD experiments and high-dimensional stochastic SGD benchmarks

---

## The Central Question

Do the "narrow valleys" (thung lũng hẹp) visible in 2D Rosenbrock contour plots accurately represent the loss landscape of an 11-million-parameter ResNet-18?

## Short Answer: No, But They Still Provide Value

### What 2D Visualizations CAN Show:
1. **Local gradient behavior** near critical points (e.g., how momentum helps escape saddle points)
2. **Qualitative optimizer differences** (e.g., Adam's adaptive step vs. SGD's fixed step)
3. **Convergence rate regimes** where analytical theory applies exactly (strongly convex, convex, non-convex)
4. **Educational intuition** for understanding optimization dynamics in controlled settings

### What 2D Visualizations CANNOT Show:
1. **High-dimensional saddle proliferation:** In d=11M dimensions, almost all critical points are saddle points (Bray & Dean 2007). 2D saddles are artificially rare.
2. **Gradient noise effects:** Neural networks use mini-batch gradients (stochastic noise σ²). 2D plots show deterministic gradients only.
3. **Overparameterization regime:** ResNet-18 has 11M parameters for 50K CIFAR10 images. This "interpolation regime" (train loss → 0) does not exist in underparameterized 2D functions.
4. **Batch normalization & residual connections:** Architectural features that fundamentally reshape the loss landscape (flatten curvature, enable deep training) cannot be visualized in 2D toy problems.
5. **Stochastic gradient noise floor:** Neural network training never reaches ||∇f|| = 0 due to mini-batch sampling noise.

---

## Research Validity Implications

This limitation does NOT invalidate the thesis work if framed correctly:

### ✅ CORRECT Framing:
> "We use 2D test functions to validate that our optimizer implementations reproduce theoretically predicted convergence rates in controlled settings where theory applies exactly (e.g., strongly convex Rosenbrock). Separately, we benchmark these optimizers on realistic neural networks (ResNet-18) to measure empirical performance under conditions (non-convexity, stochasticity, overparameterization) where 2D intuition may not transfer."

### ❌ INCORRECT Framing (Avoid):
> "Because Adam escapes narrow valleys faster than SGD on 2D Rosenbrock, it will converge faster on ResNet-18."  
> *(This is a non sequitur — 11M-dimensional landscapes have fundamentally different geometry.)*

---

## Architectural Implementation in GDSearch

### 2D Deterministic Experiments
**Files:** `src/core/test_functions.py`, `scripts/demo_saddle_point.py`  
**Characteristics:**
- Full gradient computation (no sampling noise)
- Known global optimum (e.g., (1,1) for Rosenbrock)
- Exact Hessian matrix available (2×2)
- Deterministic trajectories (reproducible without random seed)

**Valid Metrics:**
- Distance to optimum: ||x_t - x*||
- Gradient norm: ||∇f(x_t)||
- Loss regret: f(x_t) - f(x*)
- Theoretical convergence rate: O(1/k), O(1/√κ)

### High-Dimensional Stochastic Experiments
**Files:** `run_all_kaggle.py`, `src/experiments/run_nn_experiment.py`  
**Characteristics:**
- Mini-batch stochastic gradients (batch size 128)
- Unknown global optimum (non-convex landscape)
- Hessian approximation only (top eigenvalues via Lanczos)
- Stochastic trajectories (seed-dependent due to mini-batch sampling)

**Valid Metrics:**
- Training loss: f(x_t) (computed on mini-batch)
- Test accuracy: generalization performance
- Gradient noise: Var[∇f_mini-batch]
- Effective learning rate: per-parameter adaptive rates

**Invalid Metrics:**
- Distance to optimum ❌ (x* is unknown)
- Exact gradient norm ❌ (only stochastic estimates available)

---

## Visualization Constraints

### Exact Trajectory Plots (2D Only)
```python
# Valid for: Rosenbrock, Sphere, Saddle Point
plot_2d_trajectory(optimizer_path, x_star=(1, 1))  # Known optimum
```

### Projected Trajectory Plots (Neural Networks)
```python
# Valid for: ResNet-18, SimpleCNN
# Uses PCA to project 11M-dimensional path onto 2D plane
plot_pca_trajectory(optimizer_checkpoints, method='pca')
```

**⚠️ CRITICAL DISCLAIMER:**  
When presenting neural network trajectory visualizations, you MUST state:  
> "This is a 2D PCA projection of an 11-million-dimensional optimization path. The visual 'width' of the valley does not represent the true loss landscape curvature."

See [VISUALIZATION_PROJECTION_GUIDE.md](VISUALIZATION_PROJECTION_GUIDE.md) for implementation details.

---

## Theoretical Guarantees by Problem Class

| Problem Type | Gradient Type | Known Optimum? | Applicable Theory | Valid Convergence Criteria |
|--------------|---------------|----------------|-------------------|----------------------------|
| 2D Strongly Convex (Sphere) | Deterministic | ✅ Yes | O(1/√κ) for GD+Momentum | ||∇f|| < ε, f(x) - f* < δ |
| 2D Non-Convex (Rosenbrock) | Deterministic | ✅ Yes | O(1/k) for GD | ||∇f|| < ε |
| 2D Saddle Point | Deterministic | ✅ Yes | Escape time O(poly(L,λ_min,ε)) | ||∇f|| < ε |
| ResNet-18 CIFAR10 | Stochastic (batch=128) | ❌ No | None (non-convex + stochastic) | Loss plateau, test accuracy |
| SimpleMLP MNIST | Stochastic (batch=64) | ❌ No | None (non-convex + stochastic) | Loss plateau, test accuracy |

---

## Saddle Point Special Case

**File:** `scripts/demo_saddle_point.py`  
**Status:** ✅ Fully Valid

The saddle point experiment (f(x,y) = x² - y²) is a **special case where 2D intuition DOES transfer** to high-dimensional neural networks:

1. **Theoretical Foundation:** Jin et al. (2017) prove that perturbed GD escapes saddle points in polynomial time for **any dimension**.
2. **Visualization:** The 2D vector field shows the repelling/attracting directions explicitly.
3. **Neural Network Relevance:** Deep networks have exponentially many saddle points (Dauphin et al. 2014). Momentum's ability to escape saddles in 2D suggests it will also help in high-D.

**Recommendation for Thesis:** Elevate this to a **primary contribution**. It is your cleanest link between theory (provable escape time) and practice (visualized trajectories).

---

## References

- **Bray & Dean (2007):** "Statistics of critical points of Gaussian fields on large-dimensional spaces"  
  *Establishes that random high-dimensional landscapes are dominated by saddle points*

- **Dauphin et al. (2014):** "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"  
  *Shows neural network loss landscapes have exponentially many saddles*

- **Li et al. (2018):** "Visualizing the Loss Landscape of Neural Nets"  
  *Filter normalization reveals ResNet landscapes are surprisingly smooth, unlike Rosenbrock's narrow valley*

- **Jin et al. (2017):** "How to Escape Saddle Points Efficiently"  
  *Proves polynomial escape time for perturbed GD in any dimension*

---

## Action Items for Thesis Defense

### Before Submission:
1. ✅ Add this disclaimer to all neural network trajectory plots
2. ✅ Separate "Theory Validation" (2D) from "Practical Benchmarks" (NN) into distinct chapters
3. ✅ Run `scripts/check_search_budget_parity.py` and cite results in methodology

### During Defense (Anticipated Questions):

**Q:** "Why do you show 2D Rosenbrock results in a thesis about deep learning?"  
**A:** "2D experiments verify our optimizer implementations are correct by reproducing theoretical convergence rates. Neural network experiments then measure practical performance where theory does not apply. The separation is intentional and documented in Section X.Y."

**Q:** "Can you visualize the ResNet-18 loss landscape?"  
**A:** "We use PCA-projected trajectories as an approximate visualization (Figure Z). The 11M-dimensional true landscape cannot be visualized directly. We also measure spectral properties (Hessian eigenvalues) as quantitative landscape descriptors."

**Q:** "Does momentum help ResNet-18 escape saddle points like it does in 2D?"  
**A:** "Our saddle point experiment (Section 3.1) shows momentum reduces escape time by √(L/|λ_min|) in 2D. For ResNet-18, we observe momentum achieves 5% higher test accuracy (Table 4.2), but attributing this solely to saddle escape is speculative. We can definitively prove saddle escape benefit only in the controlled 2D setting."

---

## Conclusion

The 2D vs. high-dimensional separation is an **architectural strength**, not a weakness. It demonstrates:

1. **Scientific Rigor:** You validate theory where it applies exactly
2. **Practical Relevance:** You measure performance where practitioners care (real datasets)
3. **Intellectual Honesty:** You acknowledge when theory does not apply

This three-part framework (validate → measure → acknowledge limitations) is the hallmark of mature research.
