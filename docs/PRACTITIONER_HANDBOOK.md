# Optimizer Selection & Debugging Handbook

A concise, actionable guide to pick, tune, and debug optimizers across tasks.

## 1) Quick Decision Heuristics
- If training Transformers/NLP with heterogeneous gradients → start with AdamW (lr≈2e-5–5e-4)
- If training CNNs on vision (CIFAR/Imagenet) and care about generalization → try SGD+Momentum (lr≈0.01–0.1, momentum=0.9)
- If loss landscape is ill-conditioned → try Momentum or adaptive methods (RMSProp/Adam)
- If memory-constrained → prefer SGD/SGDM over Adam-family

## 2) Tuning Cheatsheet
- Learning rate (lr): the first knob; sweep log-scale (e.g., 1e-4 → 1e-1)
- Momentum β (SGDM): 0.8–0.95 typical; too high can cause oscillation
- Adam/AdamW (β1, β2): β1=0.9, β2=0.999 default; reduce β1 to 0.5–0.8 if sharp oscillations
- Weight decay: improves generalization; AdamW decouples L2 from the update
- Batch size: interacts with lr and momentum; larger batches often support higher lr

## 3) Diagnostic Symptoms → Likely Causes → Fixes
- Strong loss oscillation; grad_norm spikes
  - Cause: lr too high; momentum too high; exploding gradients
  - Fix: lower lr; lower momentum; apply gradient clipping
- Fast training but poor test accuracy
  - Cause: overfitting or adaptive methods bias
  - Fix: try SGD+Momentum, add weight decay or regularization, more augmentation
- Very slow progress; tiny grad_norm
  - Cause: lr too low or poor initialization
  - Fix: increase lr; use warmup; re-initialize
- Unstable on saddle points
  - Cause: lack of momentum or curvature adaptation
  - Fix: add momentum; try RMSProp/Adam; add Nesterov

## 4) Dynamics to Monitor
- grad_norm and update_norm over time → detect convergence and instability
- Per-layer gradient norms ("gradient heterogeneity")
- Learning-rate schedules and their effects
- Wall-clock time and peak GPU memory per run

## 5) Suggested Experiments (Ablation Ladder)
- SGD → SGD+Momentum → RMSProp → Adam → AdamW → AMSGrad
- Keep architecture and data fixed; change one mechanism at a time
- Use multi-seed; report means, std, and corrected p-values

## 6) Reproducible Templates
- MNIST (Kaggle-ready): `kaggle/mnist_publication/`
- CIFAR-10 (Kaggle-ready): `kaggle/cifar10_publication/`
- 2D dynamics: `src/experiments/run_experiment.py`

## 7) When to Switch Optimizers
- Start with AdamW when gradients are highly heterogeneous (NLP Transformers)
- Prefer SGDM when final generalization is the priority (vision)
- Consider AMSGrad if Adam is unstable; consider RMSProp for very noisy gradients

## 8) Common Pitfalls
- Comparing optimizers with mismatched budgets (epochs, time); always control for wall-clock
- Ignoring multiple comparisons; use Holm–Bonferroni
- Drawing conclusions from single-seed results

## 9) References & Further Reading
- Loshchilov & Hutter (AdamW)
- Goyal et al. (Accurate, Large Minibatch SGD)
- Ruder (An overview of gradient descent optimization algorithms)
