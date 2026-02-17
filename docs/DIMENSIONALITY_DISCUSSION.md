# Dimensionality and Landscape Transfer: Limitations of 2D Visualizations

## The Central Question

Do the "narrow valleys" visible in 2D Rosenbrock contour plots accurately represent the loss landscape of an 11-million-parameter ResNet-18?

## Short Answer: No, But They Still Provide Value

### What 2D Visualizations CAN Show:
1. **Local gradient behavior** near critical points (e.g., how momentum helps escape saddle points)
2. **Qualitative optimizer differences** (e.g., Adam's adaptive step vs. SGD's fixed step)
3. **Convergence rate regimes** where analytical theory applies exactly (strongly convex, convex, non-convex)

### What 2D Visualizations CANNOT Show:
1. **High-dimensional saddle proliferation:** In d=11M dimensions, almost all critical points are saddle points (Dauphin et al., 2014). 2D saddles are artificially rare.
2. **Gradient noise effects:** Neural networks use mini-batch gradients (stochastic noise σ²). 2D plots show deterministic gradients only.
3. **Overparameterization regime:** ResNet-18 has 11M parameters for 50K CIFAR10 images. This "interpolation regime" (train loss → 0) does not exist in underparameterized 2D functions.
4. **Batch normalization & residual connections:** Architectural features that fundamentally reshape the loss landscape (flatten curvature, enable deep training) cannot be visualized in 2D toy problems.

## Research Validity Implications

This limitation does NOT invalidate the thesis work if framed correctly:

**CORRECT Framing:**
> "We use 2D test functions to validate that our optimizer implementations reproduce theoretically predicted convergence rates in controlled settings where theory applies exactly (e.g., strongly convex Rosenbrock). Separately, we benchmark these optimizers on realistic neural networks (ResNet-18) to measure empirical performance under conditions (non-convexity, stochasticity, overparameterization) where 2D intuition may not transfer."

**INCORRECT Framing (Avoid):**
> "Because Adam escapes narrow valleys faster than SGD on 2D Rosenbrock, it will converge faster on ResNet-18."  
> *(This is a non sequitur — 11M-dimensional landscapes have fundamentally different geometry.)*

## References
- Dauphin et al. (2014): "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"
- Li et al. (2018): "Visualizing the Loss Landscape of Neural Nets" (filter normalization technique shows ResNet landscapes are surprisingly smooth, unlike Rosenbrock's narrow valley)
- Choromanska et al. (2015): "The Loss Surfaces of Multilayer Networks"
