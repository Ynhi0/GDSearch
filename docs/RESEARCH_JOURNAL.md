# Research Journal - GDSearch Project

**Purpose:** This journal documents our scientific detective work - tracking hypotheses, anomalies, insights, and the "why" behind every observation.

**Principle:** *"Every plot is not a destination, but a clue. Follow the clues."*

---

## 📅 Entry 1: November 3, 2025 - Initial Observations

### 🔍 Detective Question: Why does Adam "shoot" in early iterations?

**Observation:**
- Adam trajectories show rapid initial progress on Rosenbrock function
- SGD+Momentum takes more conservative, curved path
- See: `plots/adam_trajectory_grid_rosenbrock.png`

**Hypothesis:**
Adam's adaptive learning rates allow aggressive steps in directions with consistent gradients (the valley floor) while being cautious in high-curvature directions (valley walls). The "shooting" is actually intelligent scaling.

**Evidence to collect:**
- [ ] Per-parameter learning rate evolution
- [ ] Gradient magnitude along trajectory
- [ ] Hessian eigenvalue ratio at "shooting" points

**Connection to theory:**
Relates to adaptive moment estimation - Adam's second moment (v) tracks gradient variance, enabling larger steps where gradient is consistently large.

---

## 📅 Entry 2: Generalization Gap Anomaly

### 🔍 Detective Question: Why does SGD+Momentum generalize better despite slower convergence?

**Observation:**
- AdamW: Fast convergence, gen-gap ~0.15
- SGD+Momentum: Slower start, gen-gap ~0.08
- See: `results/summary_quantitative.csv`

**Hypothesis 1 (Sharp vs Flat Minima):**
Adam's adaptive steps allow it to settle into sharper minima (faster convergence). SGD+Momentum's momentum carries it past sharp minima, preferring flatter basins.

**Evidence collected:**
- ✅ Loss landscape plots show flatter neighborhoods for SGD+Momentum
- ✅ See: `plots/loss_landscape_*.png`

**Hypothesis 2 (Implicit Regularization):**
Momentum acts as implicit regularization by averaging gradients over time, preventing overfitting to batch-specific noise.

**Evidence to collect:**
- [ ] Per-batch gradient variance
- [ ] Trajectory stability metrics

**Practical implication:**
For production models, consider:
1. AdamW for rapid prototyping (3-5 epochs)
2. Switch to SGD+Momentum for final training (epochs 6-20)
This hybrid approach combines speed and generalization quality.

---

## 📅 Entry 3: Saddle Point Escape Behavior

### 🔍 Detective Question: How do optimizers escape saddle points?

**Observation:**
- On SaddlePoint test function, optimizers show different escape patterns
- Adam escapes quickly, SGD can get stuck
- See: Hessian eigenvalue data (λ_max × λ_min < 0 indicates saddle)

**Hypothesis:**
Momentum helps escape saddles by accumulating velocity even when gradient approaches zero at the saddle. Adam's adaptive learning may also help by amplifying noise in directions with small second moments.

**Evidence to collect:**
- [x] Hessian eigenvalue tracking (IMPLEMENTED in v2.0.0)
- [ ] Time spent near saddles (saddle_regions from analyze_saddle_escape)
- [ ] Gradient noise contribution to escape

**New experiment idea:**
Run controlled noise experiment: inject Gaussian noise with different σ values, measure escape time from saddle initialization.

---

## 📅 Entry 4: Anomaly - Seed 42 Divergence on Rosenbrock

### 🚨 ANOMALY ALERT

**Observation:**
During initial testing, SGD+Momentum with β=0.99, lr=0.01 diverged (OverflowError) on Rosenbrock function.

**Initial reaction:** Temptation to dismiss as "bad hyperparameters"

**Detective work:**
1. Why this specific combination?
2. Check initialization point: (-1.5, 2.0)
3. Compute Hessian at that point:
   - High condition number (κ >> 100)
   - Eigenvalue analysis shows one direction with very steep curvature

**Root cause:**
High momentum (0.99) + high LR (0.01) + steep local curvature = unbounded velocity accumulation

**Resolution:**
Reduced LR to 0.001 for high-momentum experiments
- See: `plots/sgdm_trajectory_series_rosenbrock.png` (stable with lr=0.001)

**Insight gained:**
Momentum amplifies not just gradient signal but also instabilities. High momentum requires proportionally lower learning rate, especially on ill-conditioned functions.

**Practical implication:**
Rule of thumb: If β > 0.9, use lr < 0.001 * (1-β) as starting point.

---

## 📅 Entry 5: Per-Layer Gradient Distribution

### 🔍 Detective Question: Why do gradient norms differ across layers?

**Observation:**
- AdamW shows more uniform gradient distribution across layers
- SGD+Momentum emphasizes later layers early on
- See: `plots/*_layer_grads.png`

**Hypothesis:**
Later layers (closer to loss) receive stronger error signals initially. AdamW's per-parameter adaptation normalizes this imbalance. SGD+Momentum propagates raw gradients, maintaining the natural imbalance.

**Evidence collected:**
- ✅ Bar charts at epochs [1, 10, 20]
- Pattern: AdamW stays uniform, SGD+Momentum balances over time

**Connection to practice:**
This explains why AdamW works well out-of-the-box, while SGD+Momentum may need:
- Layer-wise learning rates
- Gradient clipping
- Longer warmup

**Further investigation needed:**
- [ ] Gradient flow analysis (gradient magnitude at each layer)
- [ ] Compare with layer normalization / batch normalization effects

---

## 🔬 Methodology Notes

### When to Add Journal Entries

1. **After every experiment run:** Record what you expected vs what you observed
2. **When you spot an anomaly:** Document it immediately, resist the urge to dismiss
3. **When you form a hypothesis:** Write it down explicitly
4. **When you find evidence:** Link it back to hypotheses

### Question Templates

Use these to guide your detective work:

1. **Observation:** "What exactly did I see?"
2. **Hypothesis:** "Why might this happen?"
3. **Evidence:** "What would prove/disprove this?"
4. **Implications:** "So what? Why does this matter?"
5. **Next steps:** "What should I investigate next?"

---

## 🎯 Active Investigations

### Priority 1: High Impact
- [ ] Sensitivity analysis around best hyperparameters (lr±10%)
- [ ] Basin of attraction mapping (grid of initializations)
- [ ] Controlled noise experiments (gradient perturbation)

### Priority 2: Theoretical Validation
- [ ] Eigenvalue spectrum evolution during training
- [ ] Gradient noise vs convergence speed correlation
- [ ] Loss landscape curvature along training trajectory

### Priority 3: Practical Applications
- [ ] Hybrid optimizer strategy (Adam → SGD+Momentum)
- [ ] Learning rate scheduling based on landscape features
- [ ] Early stopping criteria based on generalization gap

---

## 📚 Connections to Literature

### Papers to reference:
1. **Sharp minima and generalization:**
   - "Sharp Minima Can Generalize For Deep Nets" (Dinh et al., 2017)
   - Need to discuss our loss landscape findings in this context

2. **Adaptive methods:**
   - "On the Convergence of Adam and Beyond" (Reddi et al., 2018)
   - Our observation about Adam's early speed aligns with their analysis

3. **Momentum and saddle escapes:**
   - "How Does Batch Normalization Help Optimization?" (Santurkar et al., 2018)
   - Discusses landscape smoothing - relevant to our momentum findings

---

## 💡 Eureka Moments

> "The anomaly where SGD+Momentum diverged wasn't a failure - it was a window into understanding the interaction between momentum, learning rate, and local curvature. This single 'failed' experiment taught us more than 10 successful ones."

> "By treating generalization gap not as a final metric but as a clue, we discovered the sharp vs flat minima story - connecting 2D toy functions to real neural network behavior."

---

## 🔄 Iteration Log

### v1.0 → v2.0
- Added Hessian eigenvalue tracking (enables saddle analysis)
- Implemented convergence detection (quantifies "speed")
- Enhanced documentation (from technical → detective mindset)

### Future iterations:
- Add per-parameter learning rate tracking for Adam
- Implement basin-of-attraction visualization
- Add automated anomaly detection in training logs

---

**Last Updated:** November 3, 2025
**Status:** Active investigation
**Next Review:** After sensitivity analysis completion
