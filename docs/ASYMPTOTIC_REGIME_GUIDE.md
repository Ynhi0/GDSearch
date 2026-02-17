# When Does Asymptotic Theory Apply? A Practical Guide

## The Fundamental Mismatch

**Optimization Theory:** Analyzes behavior as iterations → ∞  
**Deep Learning Practice:** Stop training after 20-200 epochs

**Consequence:** Fitting theoretical O(1/k) curves to ResNet-18 training (which stops at epoch 50) is often mathematically invalid.

## Valid vs. Invalid Use Cases

### ✅ VALID: 2D Test Functions (Rosenbrock, Quadratic)

**Why It Works:**
1. **Known minimum:** f* = 0 (exactly)
2. **Can run until convergence:** Iterate until ||grad|| < 1e-8
3. **Reaches asymptotic regime:** Loss actually follows f(t) - 0 ≈ A/t^α for large t

**Example Code:**
```python
# 2D Rosenbrock with GD
result = run_2d_experiment('rosenbrock', optimizer='SGD', max_iters=10000, tol=1e-8)
rate = compute_empirical_rate(result['losses'], known_min=0.0)
# Output: alpha=1.02 (close to theoretical O(1/k) = alpha=1)
```

**Thesis Use:**  
"We validated that our SGD implementation achieves the theoretical O(1/k) convergence rate on strongly convex 2D quadratics, as shown by the power-law fit exponent α=1.02±0.05."

### ❌ INVALID: ResNet-18 on CIFAR10 (Stopped at Epoch 50)

**Why It Fails:**
1. **Unknown minimum:** f* is unknown (non-convex landscape)
2. **Early stopping:** Training stops before reaching ANY asymptotic regime
3. **Scheduler interference:** Learning rate decays (CosineAnnealing, StepLR) invalidate constant-step-size theory
4. **Stochastic noise:** Mini-batch variance prevents exact convergence (noise floor)

**Example of WRONG Analysis:**
```python
# ResNet-18 training stopped at epoch 50
resnet_losses = load_training_log('resnet18_cifar10.csv')['train# WRONG CONCLUSION: "ResNet training is slower than theory predicts"
# CORRECT CONCLUSION: "ResNet training never reached asymptotic regime — stopped in transient phase"
```

### 🔶 PARTIAL VALIDITY: Neural Networks with Specific Conditions

**When Theory Might Apply:**
1. **Interpolation regime:** Train until loss < 0.01 (near zero)
2. **Fixed LR:** No scheduler (constant step size throughout)
3. **Large batch:** Reduce stochastic noise (batch_size ≥ 1024)
4. **Sufficient iterations:** Run 200+ epochs (not 20-50)

**Example Code:**
```python
# Modified training to match theory assumptions
config = {
    'epochs': 500,  # Much longer than typical
    'scheduler': None,  # Fixed LR (no decay)
    'batch_size': 2048,  # Large batch (low noise)
    'early_stopping': False  # Force full run
}
result = train_resnet18(config)
# NOW fitting asymptotic rates is valid (if interpolation is reached)
```

**Thesis Use:**  
"Under controlled conditions (fixed LR, large batch, extended training), ResNet-18 training loss follows a power-law decay with exponent α=0.8, consistent with sub-linear convergence in the non-convex setting."

## Recommended Analysis Strategy

**For 2D Functions:**
- ✅ Fit asymptotic rates (O(1/k), O(1/√κ))
- ✅ Compare to theoretical bounds
- ✅ Claim validation of optimizer implementation

**For Neural Networks:**
- ❌ Do NOT fit asymptotic curves to early-stopped training
- ✅ Report empirical speed: "Time to reach loss < X"
- ✅ Report final metrics: "Test accuracy at epoch 50"
- ✅ Use gradient norm bounds (non-convex theory): E[||∇f||²] ≤ C/√T

**Example Correct Statement:**
> "While asymptotic theory predicts O(1/k) convergence for strongly convex problems, our ResNet-18 experiments measure empirical convergence speed: SGD reaches train_loss=0.1 in 35 epochs, while Adam requires only 18 epochs. This 2× speedup is consistent with Adam's adaptive learning rate advantage in the finite-time regime."

## References
- Bottou et al. (2018): "Optimization Methods for Large-Scale Machine Learning" (Section 4.3: Finite-time vs. asymptotic analysis)
- Jain et al. (2017): "Parallelizing Stochastic Gradient Descent for Least Squares Regression: Mini-batching, Averaging, and Model Misspecification" (noise floor analysis)
- Ghadimi & Lan (2013): "Stochastic First- and Zeroth-order Methods for Nonconvex Stochastic Programming" (non-convex finite-time bounds)
