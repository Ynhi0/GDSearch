# CRITICAL GAPS & FIXES - Research Proposal Alignment

**Date**: December 7, 2025  
**Status**: 🚨 CRITICAL ISSUES IDENTIFIED

---

## Vietnamese Research Proposal Requirements Analysis

### Core Research Objectives (From Proposal)

The proposal explicitly requires:

1. **Theoretical Analysis** (Phần lý thuyết):
   - ✅ Systematic review of convergence rate results
   - ✅ L-smoothness and PL condition analysis
   - ✅ Theoretical bounds implementation (`src/analysis/theoretical_bounds.py`)

2. **Experimental Analysis** (Phần thực nghiệm):
   - ✅ Algorithm implementation (GD, SGD, Momentum, Adam, RMSprop)
   - ⚠️  **PARTIAL**: Hyperparameter sensitivity analysis (β, β1, β2)
   - ⚠️  **PARTIAL**: Trajectory visualization and dynamics analysis
   - ❌ **MISSING**: Integration of trajectory analysis with training experiments

3. **Comparative Dynamics Analysis** (Phân tích động học so sánh):
   - ✅ Basic comparison exists
   - ❌ **MISSING**: "khảo sát, trực quan hóa, lý giải cách thức các siêu tham số đặc trưng (β của Momentum; β1, β2 của Adam) định hình quỹ đạo và hành vi hội tụ từng bước"
   - ❌ **MISSING**: Instantaneous rate analysis ("tốc độ tức thời")
   - ❌ **MISSING**: Oscillation/fluctuation quantification ("dao động")

4. **Non-Convex Landscape Focus** (Hàm phi lồi):
   - ✅ 2D test functions (Rosenbrock, Rastrigin, Ackley)
   - ✅ Real neural network training
   - ⚠️  **PARTIAL**: Loss landscape visualization (exists but not integrated with all experiments)

---

## CRITICAL GAPS IDENTIFIED

### GAP #1: Beta Parameter Dynamics NOT Properly Analyzed

**Proposal Requirement**:
> "khảo sát hệ thống và trực quan hóa ảnh hưởng của các siêu tham số đặc trưng (β, β1, β2) lên các khía cạnh động học như quỹ đạo, tốc độ tức thời và độ ổn định"

**Current State**:
- ✅ `hyperparameter_sensitivity.py` does β sweeps
- ❌ But it ONLY outputs CSV files - no trajectory plots
- ❌ No "instantaneous rate" analysis
- ❌ No "oscillation index" visualization

**Required Fix**:
```python
# MUST ADD to hyperparameter_sensitivity.py:
def visualize_beta_trajectories(results_df, output_dir):
    """
    Create trajectory plots showing how β affects:
    1. Path taken (2D contour + trajectory overlay)
    2. Instantaneous velocity (speed per iteration)
    3. Oscillation magnitude (deviation from smooth path)
    """
    # Create publication-quality plots with:
    # - Trajectory overlay on loss landscape
    # - Instantaneous gradient norm plot
    # - Oscillation/smoothness metric plot
```

### GAP #2: Dynamics Analysis for REAL Training (Not Just 2D)

**Proposal Requirement**:
> "phân tích chi tiết các đặc tính động học so sánh (độ mượt - smoothness, tốc độ tức thời - instantaneous rate/update magnitude, dao động - oscillations/fluctuations)"

**Current State**:
- ✅ 2D trajectory visualization exists (`trajectory_2d.py`)
- ❌ **CRITICAL**: No trajectory/dynamics analysis for MNIST, CIFAR-10, ResNet, NLP training
- ❌ No "update magnitude" tracking during training
- ❌ No "oscillation quantification" for real training

**Required Fix**:
Need to add `DynamicsMonitor` class that tracks during training:
```python
class DynamicsMonitor:
    """Track training dynamics for proposal compliance"""
    def __init__(self):
        self.grad_norms = []
        self.param_updates = []
        self.loss_oscillations = []
    
    def track_step(self, optimizer, loss):
        # Compute instantaneous metrics
        grad_norm = compute_total_grad_norm(optimizer)
        update_magnitude = compute_update_magnitude(optimizer)
        
        # Track oscillation (deviation from EMA)
        oscillation = abs(loss - self.loss_ema)
        
        # Store for analysis
        self.grad_norms.append(grad_norm)
        self.param_updates.append(update_magnitude)
        self.loss_oscillations.append(oscillation)
```

### GAP #3: Theory-Practice Convergence Comparison INCOMPLETE

**Proposal Requirement**:
> "đối chiếu tốc độ hội tụ quan sát được với các dự đoán lý thuyết"

**Current State**:
- ✅ `convergence_rate_validation.py` exists
- ❌ But it ONLY runs on test functions (not real training)
- ❌ No integration with actual MNIST/CIFAR/ResNet training results
- ❌ No "theoretical bound overlay" on training curves

**Required Fix**:
```python
def compare_theory_vs_practice(training_results_csv, optimizer_name):
    """
    Load actual training results and overlay theoretical bounds:
    1. Load training loss/gradient curve
    2. Compute theoretical O(1/k) or O(ρ^k) bound
    3. Plot both on same graph with shaded confidence region
    4. Calculate deviation percentage
    """
```

### GAP #4: Ablation Studies Missing Dynamics Visualization

**Proposal Requirement**:
All ablation studies should show "how mechanisms affect dynamics" not just final metrics.

**Current State**:
- ✅ Ablation studies exist with visualizations
- ⚠️  Visualizations show FINAL metrics only (bar charts, box plots)
- ❌ No "per-iteration dynamics comparison" plots

**Required Fix**:
Add to `ablation_plots.py`:
```python
def create_ablation_dynamics_plot(results_df, output_path):
    """
    Show HOW ablated feature affects training dynamics:
    - Loss curve comparison (baseline vs feature enabled)
    - Gradient norm evolution
    - Update magnitude over time
    """
```

### GAP #5: Missing Experiments from Proposal Scope

**Proposal states**: "ưu tiên sử dụng các hàm kiểm tra tổng hợp phi lồi 2 chiều (2D synthetic non-convex test functions)"

**Current State**:
- ✅ Rosenbrock (2D)
- ✅ Rastrigin (2D)
- ❌ **MISSING**: Ackley 2D visualization (exists as function but not visualized with trajectories)
- ❌ **MISSING**: "Thung lũng hẹp" (narrow valley) test function
- ❌ **MISSING**: "Điều kiện yếu" (weak condition / ill-conditioned) test function

**Required Fix**:
Add to `src/core/test_functions.py`:
```python
class BealeFunction:
    """Narrow valley test function (ill-conditioned)"""
    @staticmethod
    def __call__(x):
        # Beale function: f(x,y) = (1.5 - x + xy)^2 + ...
        # Known for narrow curved valley
        
class StyblinskiTang:
    """Multi-modal test function with weak local minima"""
```

---

## MISSING LOGIC IN EXISTING CODE

### Issue #1: `run_all_kaggle.py` - Medical Experiment Parameter Mismatch

**Line 2796**: `run_medical_experiment` signature has `skip_tuning` parameter but it's NOT used inside function.

**Fix Required**:
```python
# Line ~2850 in run_medical_experiment
# CURRENT: Always runs tuning
# FIX: Respect skip_tuning parameter
if not skip_tuning:
    # Run Optuna tuning
else:
    # Use default hyperparameters
```

### Issue #2: `hyperparam_sensitivity` NOT Integrated in run_all_kaggle.py

**Lines 6096-6122**: Hyperparameter sensitivity is in the main() but:
- ❌ It only runs on test functions (rosenbrock, ackley)
- ❌ It does NOT integrate with ACTUAL training experiments
- ❌ No beta sweep for MNIST/CIFAR/ResNet training

**Fix Required**:
```python
# ADD after hyperparam_sensitivity:
def run_training_beta_sensitivity(dataset='MNIST', output_dir='results/beta_sensitivity'):
    """
    Run actual neural network training with different β values.
    Required by proposal: show β effect on REAL training dynamics.
    """
    for beta in [0.0, 0.5, 0.9, 0.99]:
        # Train MNIST with SGD Momentum using this beta
        # Track loss, grad_norm, update_magnitude per iteration
        # Save for dynamics visualization
```

### Issue #3: Resume Logic Missing for Analysis Experiments

**Lines 6096+**: The NEW experiments (hyperparam_sensitivity, convergence_validation, ablation_comprehensive, 2d_visualization) do NOT have resume parameter:

```python
# CURRENT - NO RESUME:
momentum_beta_sweep(...) # Missing resume check

# SHOULD BE:
if not resume or not os.path.exists(output_file):
    momentum_beta_sweep(...)
```

### Issue #4: Kaggle Notebook Cell 12 Outdated

The Kaggle notebook Quick Access Guide (Cell 12) does NOT mention:
- ❌ `hyperparam_sensitivity/` directory
- ❌ `convergence_validation/` directory
- ❌ `ablation_comprehensive/` directory
- ❌ `2d_visualization/` directory

---

## BUGS IDENTIFIED

### BUG #1: `run_initialization_ablation` - Results Directory Not Passed

**Line 4594** in `run_all_kaggle.py`:
```python
def run_initialization_ablation(device='cuda', epochs=10, seeds=[1,2,3,4,5], quick=False):
    # ...
    from src.experiments.initialization_ablation import run_initialization_ablation
    # ...
    df = run_initialization_ablation(...)  # Missing results_dir parameter
```

**Impact**: Results are saved to default location, not `experiments/init_ablation/`

**Fix**:
```python
def run_initialization_ablation(device='cuda', epochs=10, seeds=[1,2,3,4,5], quick=False, results_dir=None):
    # Pass results_dir to underlying function
```

### BUG #2: Convergence Validation - Division by Zero Risk

**File**: `src/experiments/convergence_rate_validation.py`

**Issue**: When fitting convergence rate, if `iterations[0] == 0`, log(iterations) will fail.

**Fix**: Add safety check before curve fitting.

### BUG #3: Trajectory 2D - Memory Leak for Large Iterations

**File**: `src/visualization/trajectory_2d.py`

**Issue**: Stores ALL iterations in memory for plotting. For 10,000+ iterations, this can cause OOM.

**Fix**: Add downsampling for visualization.

---

## REQUIRED ADDITIONS TO MATCH PROPOSAL

### Addition #1: Instantaneous Metrics Module

**New File**: `src/analysis/dynamics_metrics.py`

```python
def compute_instantaneous_speed(trajectory):
    """Compute ||x_t - x_{t-1}|| per iteration"""
    
def compute_smoothness_index(trajectory):
    """Measure trajectory curvature (angle changes)"""
    
def compute_oscillation_magnitude(loss_history):
    """Quantify deviation from exponential moving average"""
```

### Addition #2: Theory-Practice Integration Script

**New File**: `src/analysis/theory_practice_comparison.py`

```python
def overlay_theoretical_bounds(training_csv, optimizer, L, mu=None):
    """
    Load training results and overlay theoretical convergence rate:
    - Convex: O(1/k) or O(exp(-k/κ))
    - PL: O(exp(-μk/L))
    - General: O(1/√k)
    """
```

### Addition #3: Training Dynamics Tracker

**New File**: `src/core/dynamics_tracker.py`

```python
class TrainingDynamicsTracker:
    """
    Track per-iteration dynamics during REAL training.
    Required by proposal for dynamics analysis.
    """
    def __init__(self):
        self.iterations = []
        self.losses = []
        self.grad_norms = []
        self.update_magnitudes = []
        self.param_distances = []  # Distance from init
        
    def log_step(self, iteration, loss, model, optimizer):
        # Compute and store all dynamics metrics
        
    def save_dynamics(self, output_path):
        # Save to CSV for post-hoc analysis
        
    def plot_dynamics(self, output_dir):
        # Create dynamics visualization plots
```

---

## ACADEMIC RIGOR ISSUES

### Issue #1: Statistical Tests Not Applied to ALL Comparisons

Some ablation studies use basic mean comparison without t-tests or effect sizes.

**Fix**: Enforce `compare_two_optimizers()` from `statistical_analysis.py` everywhere.

### Issue #2: Multi-Seed Results Not Always Aggregated Properly

Some experiments save per-seed CSVs but don't create aggregated statistics.

**Fix**: Add post-processing step that computes mean ± std across seeds for ALL experiments.

### Issue #3: Publication Claims Require More Rigorous Validation

Proposal aims for academic publication. Current code lacks:
- ❌ Confidence intervals on all plots
- ❌ Power analysis reporting (exists but not in all ablations)
- ❌ Multiple comparison correction (exists but not applied everywhere)

---

## ACTION PLAN

### Priority 1 (CRITICAL - Required by Proposal):
1. ✅ Add dynamics tracking to MNIST/CIFAR/ResNet training
2. ✅ Integrate β parameter sweeps with REAL training (not just test functions)
3. ✅ Add trajectory visualization for actual training (not just 2D)
4. ✅ Create theory-practice convergence comparison plots

### Priority 2 (HIGH - Improves Academic Quality):
5. ✅ Add instantaneous metrics computation module
6. ✅ Fix resume logic for all new experiments
7. ✅ Add missing test functions (Beale, Styblinski-Tang)
8. ✅ Update Kaggle notebook Cell 12

### Priority 3 (MEDIUM - Bug Fixes):
9. ✅ Fix `run_initialization_ablation` results_dir bug
10. ✅ Add division-by-zero protection in convergence validation
11. ✅ Add memory-efficient trajectory plotting
12. ✅ Enforce statistical tests everywhere

### Priority 4 (LOW - Polish):
13. Remove unused imports
14. Add docstring completeness check
15. Verify all CSV column names consistent

---

## FILES TO CREATE

1. `src/analysis/dynamics_metrics.py` - Instantaneous metrics
2. `src/analysis/theory_practice_comparison.py` - Theory vs practice
3. `src/core/dynamics_tracker.py` - Real-time tracking
4. `src/experiments/training_beta_sensitivity.py` - Beta sweeps on REAL training

## FILES TO MODIFY

1. `run_all_kaggle.py` - Add dynamics tracking, fix bugs
2. `src/experiments/hyperparameter_sensitivity.py` - Add visualization
3. `src/experiments/convergence_rate_validation.py` - Fix div-by-zero
4. `src/visualization/trajectory_2d.py` - Add memory efficiency
5. `src/experiments/initialization_ablation.py` - Fix results_dir
6. `kaggle/run_benchmark.ipynb` - Update Cell 12

---

**Status**: Ready for implementation
**Estimated Time**: 4-6 hours for full implementation
**Academic Impact**: HIGH - Directly addresses proposal requirements
