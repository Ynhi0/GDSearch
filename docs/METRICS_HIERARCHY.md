# Metrics Hierarchy for Convergence Analysis

## The Logical Flaw

**Proposal Claim:** "We analyze convergence rate of optimization algorithms."  
**Common Mistake:** Showing graphs of "Test Accuracy" as primary evidence.

**Why This is Wrong:**
- **Convergence Rate** (optimization theory) = How fast TRAINING LOSS → minimum
- **Generalization** (learning theory) = Gap between TRAIN and TEST performance

These are INDEPENDENT properties:
- An algorithm can have **slow convergence** but **good generalization** (e.g., SGD+Momentum with small batch)
- An algorithm can have **fast convergence** but **poor generalization** (e.g., Adam with large batch)

## Correct Metrics for Each Research Question

### Research Question 1: "What is the convergence rate?"
**Primary Metric:** Training Loss vs. Iterations  
**Secondary Metric:** Gradient Norm (for non-convex)  
**Irrelevant Metric:** Test Accuracy (measures generalization, not convergence)

**Example Correct Statement:**
> "SGD with momentum converges to training loss < 0.01 in 25 epochs, while vanilla SGD requires 40 epochs."

**Example INCORRECT Statement:**
> "SGD with momentum has better convergence because its test accuracy is 2% higher."  
> *(Higher test accuracy means better generalization, NOT faster convergence.)*

### Research Question 2: "Which optimizer generalizes better?"
**Primary Metric:** Generalization Gap = Test Loss - Train Loss  
**Secondary Metric:** Test Accuracy (at SAME training loss level)  
**Control Variable:** Ensure all optimizers reach same training loss before comparing

**Example Correct Statement:**
> "When both optimizers reach train_loss=0.05, Adam has gen_gap=0.12 while Momentum has gen_gap=0.08, suggesting Momentum finds flatter minima."

###Research Question 3: "Overall practical performance?"
**Primary Metric:** Test Accuracy at fixed compute budget (e.g., 50 epochs)  
**Justification:** This is the metric practitioners care about (but it conflates convergence + generalization + hyperparameter tuning quality)

## Implications for Thesis

**CORRECT Thesis Structure:**

**Chapter 4: Convergence Rate Analysis**  
- Metric: Training Loss vs. Time/Iterations  
- Figures: Training loss curves (NOT test accuracy)  
- Theory comparison: O(1/k) vs O(1/√κ) fits on TRAINING LOSS  

**Chapter 5: Generalization Analysis**  
- Metric: Generalization Gap, Sharpness (Hessian eigenvalues)  
- Figures: Train vs Test loss, flatness metrics  
- Theory: PAC bounds, uniform stability  

**Chapter 6: Practical Benchmarks**  
- Metric: Final Test Accuracy (at fixed compute budget)  
- Figures: Test accuracy curves, optimizer comparison tables  
- Discussion: "Test accuracy reflects BOTH convergence speed AND generalization quality"

**INCORRECT Structure (Avoid):**

**Chapter 4: Optimizer Comparison**  
- Metric: Test Accuracy (conflates convergence + generalization)  
- Figures: Bar charts of test accuracy  
- Conclusion: "Adam is faster because test accuracy is higher" ← LOGICAL ERROR
