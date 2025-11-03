# GDSearch - Optimizer Dynamics Research Platform

A comprehensive Python framework for comparing gradient descent algorithms on 2D test functions and neural networks (MNIST/CIFAR-10). Features systematic hyperparameter tuning, convergence analysis, curvature tracking, loss landscape visualization, **multi-seed experiments**, and **statistical analysis**.

## 🎯 Features

### Core Capabilities
- ✅ **4 Optimization Algorithms:** SGD, SGD+Momentum, RMSProp, Adam/AdamW
- ✅ **3 2D Test Functions:** Rosenbrock, Ill-Conditioned Quadratic, Saddle Point
- ✅ **Neural Networks:** SimpleMLP (MNIST), SimpleCNN/ConvNet (CIFAR-10) with PyTorch
- ✅ **Systematic Hyperparameter Tuning:** Two-stage pipeline (LR sweep → parameter sweep)
- ✅ **Convergence Detection:** Dual conditions (grad norm threshold OR loss delta)
- ✅ **Advanced Analysis:**
  - Hessian eigenvalue tracking (λ_min, λ_max, condition number)
  - Loss landscape 1D/2D visualization
  - Per-layer gradient norms
  - Curvature analysis (trajectory turning angles)
  - Generalization gap monitoring

### 🆕 Scientific Rigor (NEW!)
- ✅ **Multi-Seed Experiments:** Run experiments with multiple random seeds for statistical reliability
- ✅ **Statistical Analysis:** T-tests, effect sizes (Cohen's d), 95% confidence intervals
- ✅ **Error Bar Visualization:** Plots with mean ± std bands
- ✅ **Unit Tests:** 35 tests verifying gradients and optimizer correctness (pytest)
- ✅ **Input Validation:** Comprehensive error checking and input sanitization
- ✅ **Ablation Studies:** Component-wise isolation to quantify contributions
- ✅ **Baseline Comparisons:** Compare custom implementations with PyTorch built-ins

## 📁 Project Structure

```
GDSearch/
├── configs/
│   ├── nn_tuning.json          # MNIST hyperparameter sweeps
│   └── cifar10_tuning.json     # CIFAR-10 configurations
├── results/                    # CSV outputs (experiments, summaries)
├── plots/                      # All visualizations (PNG)
├── data/                       # Dataset utilities
│
├── Core Modules:
│   ├── test_functions.py       # 2D test functions with analytic derivatives
│   ├── optimizers.py           # Optimizer implementations
│   ├── models.py               # PyTorch NN models (MLP, CNN, ConvNet)
│   ├── data_utils.py           # MNIST/CIFAR-10 loaders
│   ├── validation.py           # Input validation & error handling
│
├── Experiment Runners:
│   ├── run_experiment.py       # 2D experiments with Hessian tracking
│   ├── run_nn_experiment.py    # NN training with convergence detection
│   ├── tune_nn.py              # Two-stage hyperparameter tuning
│   ├── run_loss_landscape.py   # Loss landscape visualization
│   ├── run_multi_seed.py       # Multi-seed experiment framework 🆕
│   ├── run_full_analysis.py    # Complete pipeline: experiments → stats → plots 🆕
│   ├── run_ablation_study.py   # Component-wise ablation 🆕
│   ├── run_baseline_comparison.py  # Compare with PyTorch optimizers 🆕
│
├── Analysis & Visualization:
│   ├── plot_results.py         # Comprehensive plotting (now with error bars!)
│   ├── plot_eigenvalues.py     # Hessian eigenvalue visualization
│   ├── loss_landscape.py       # Loss surface probing
│   ├── generate_summaries.py   # Quantitative & qualitative tables
│   ├── statistical_analysis.py # T-tests, effect sizes, confidence intervals 🆕
│
├── Testing & Validation:
│   ├── tests/
│   │   ├── test_gradients.py   # Numerical gradient verification 🆕
│   │   └── test_optimizers.py  # Optimizer correctness tests 🆕
│
├── Documentation:
│   ├── README.md               # This file
│   ├── MULTISEED_GUIDE.md      # Guide for multi-seed experiments 🆕
│   ├── LIMITATIONS.md          # Known limitations & assumptions 🆕
│   ├── IMPROVEMENT_PROGRESS.md # Progress tracking 🆕
│   └── CRITICAL_VALIDATION_REPORT.md  # Scientific validation report
│
├── Automation:
│   └── run_all.py              # Complete reproducibility pipeline
│   └── nn_workflow.py          # Quick demo workflow
│
├── Documentation:
│   ├── REPORT.md               # Synthesis report with ablation study
│   ├── hypothesis_matrix.md    # Theory ⇄ Experiment mapping
│   └── README_PROJECT.md       # Detailed technical documentation
│
└── requirements.txt            # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project
cd /workspaces/GDSearch

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, matplotlib, scipy, torch, torchvision, tqdm, pytest

### Running Tests (Verify Installation)

```bash
# Run all tests (gradients + optimizers)
pytest tests/ -v

# Expected: 35 tests passed ✅
```

### Running Experiments

#### Option 1: Multi-Seed Statistical Analysis (Recommended) 🆕
```bash
# Full pipeline: experiments → aggregation → stats → plots
python run_full_analysis.py \
    --config configs/nn_tuning.json \
    --seeds 1,2,3,4,5 \
    --compare AdamW-SGDMomentum,Adam-RMSProp

# Quick test with 3 seeds
python run_full_analysis.py --seeds 1,2,3
```

**Output:**
- Multi-seed results with mean ± std
- Statistical comparisons (t-tests, p-values, effect sizes)
- Plots with error bars
- Aggregated JSON summary

#### Option 2: Ablation Study (Component Analysis) 🆕
```bash
# Test each optimizer component in isolation
python run_ablation_study.py

# Components tested:
#   1. SGD baseline (no momentum, no adaptive LR)
#   2. SGD + Momentum
#   3. RMSProp (adaptive LR only)
#   4. Adam (full)
#   5. Adam with L2 regularization
#   6. AdamW (decoupled weight decay)
```

#### Option 3: Baseline Comparison 🆕
```bash
# Compare custom implementations with PyTorch built-ins
python run_baseline_comparison.py

# Compares:
#   - Custom Adam vs torch.optim.Adam
#   - Custom AdamW vs torch.optim.AdamW
#   - Custom SGD+Momentum vs torch.optim.SGD
#   - Custom RMSProp vs torch.optim.RMSProp
```

#### Option 4: Traditional Single-Seed Pipeline
```bash
# Run everything: 2D + NN tuning + summaries + plots
python run_all.py

# Skip specific phases
python run_all.py --skip-2d              # Skip 2D experiments
python run_all.py --skip-tuning          # Skip NN hyperparameter tuning
python run_all.py --summaries-only       # Only regenerate summaries

# Quick mode (reduced iterations)
python run_all.py --quick
```

#### Option 5: Step-by-Step (Learning Mode)

```bash
# 1. Run 2D test function experiments (with Hessian eigenvalue tracking)
python run_experiment.py

# 2. Run neural network hyperparameter tuning
python tune_nn.py

# 3. Generate loss landscape visualizations
python run_loss_landscape.py

# 4. Create summary tables and plots
python generate_summaries.py

# 5. Visualize Hessian eigenvalues (optional)
python plot_eigenvalues.py
```

#### Option 6: Quick Demo
```bash
# Run short MNIST demo (2 epochs)
python nn_workflow.py
```

## 📊 Key Outputs

### Results Directory (`results/`)
- **Experiment CSVs:** `NN_<model>_<dataset>_<optimizer>_lr<lr>_seed<seed>[_tag].csv`
  - Tags: `sweepLR`, `sweepWD`, `sweepMOM`, `final`
- **Summary Tables:**
  - `summary_quantitative.csv`: Final metrics, convergence iters/time
  - `summary_qualitative.csv/md`: Smoothness, oscillation ratings

### Plots Directory (`plots/`)

**2D Visualizations:**
- `*_trajectory.png`: Optimization paths with contours
- `*_eigenvalues.png`: λ_min, λ_max, condition number evolution
- `dynamics_triplet_*.png`: Update/grad/curvature vs iteration
- `adam_trajectory_grid_*.png`: β1×β2 hyperparameter grid
- `sgdm_trajectory_series_*.png`: Momentum sweep (β values)
- `trajectory_3d_*.png`: 3D trajectory on function surface

**Neural Network Visualizations:**
- `*_gen_gap.png`: Generalization gap + test accuracy (dual y-axis)
- `*_layer_grads.png`: Per-layer gradient norms at epochs [1, 10, 20]
- `loss_landscape_1d.png`: 1D loss slice along random direction
- `loss_landscape_2d_surface.png`: 2D loss surface around trained weights
- `loss_landscape_2d_contour.png`: Contour map of loss landscape

## 📖 Understanding the Outputs

### Convergence Detection
The system automatically detects convergence using dual conditions:
- **Condition 1:** `grad_norm < 1e-6`
- **Condition 2:** `abs(loss[t] - loss[t-200]) < 1e-7` (windowed loss delta)

When convergence is detected, a `meta` row is logged with `(global_step, time_sec)`.

### Hessian Eigenvalue Interpretation
- **λ_max, λ_min:** Largest and smallest curvatures
- **Condition number (κ = |λ_max / λ_min|):** Measures local ill-conditioning
- **Eigenvalue product (λ_max × λ_min):**
  - `> 0`: Locally convex (both eigenvalues same sign)
  - `< 0`: Saddle point (eigenvalues opposite signs)

### Generalization Gap
`gen_gap = test_loss - train_loss`

Smaller gap indicates better generalization. Our findings:
- **AdamW:** Fast convergence but larger gen-gap (~0.15)
- **SGD+Momentum:** Slower start but smaller gen-gap (~0.08), better generalization

## 🔬 Advanced Usage

### Custom Hyperparameter Tuning

Edit `configs/nn_tuning.json`:
```json
{
  "dataset": "MNIST",
  "model": "SimpleMLP",
  "sweeps": [
    {
      "optimizer": "AdamW",
      "lr_values": [0.1, 0.01, 0.001, 0.0001],
      "weight_decay_values": [0.0, 0.0001, 0.0005],
      "epochs": 3
    }
  ],
  "final": {
    "epochs": 20,
    "capture_layer_grad_epochs": [1, 10, 20]
  },
  "convergence": {
    "grad_norm_threshold": 1e-6,
    "loss_delta_threshold": 1e-7,
    "loss_window": 200
  }
}
```

Then run: `python tune_nn.py`

### Adding Custom Test Functions

```python
# In test_functions.py
class MyFunction(TestFunction):
    def compute(self, x, y):
        return x**2 + y**2  # Your function here
    
    def gradient(self, x, y):
        return 2*x, 2*y  # Analytic gradient
    
    def hessian(self, x, y):
        return np.array([[2, 0], [0, 2]])  # Analytic Hessian
```

### Adding Custom Optimizers

```python
# In optimizers.py
class MyOptimizer(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.state = {}  # Internal state
    
    def step(self, params, gradients):
        # Your update rule here
        new_params = ...
        return new_params
    
    def reset(self):
        self.state = {}
```

## 📈 Results & Insights

### Ablation Study: Optimizer Comparison

From `REPORT.md`:

| Optimizer | MNIST Test Acc | Gen Gap | Convergence Speed | Landscape |
|-----------|----------------|---------|-------------------|-----------|
| **AdamW** | ~97.5% | ~0.15 | Fast (early epochs) | Sharper minima |
| **SGD+Momentum** | ~97.6% | ~0.08 | Slower start | Flatter minima |

**Key Takeaway:** Start with AdamW for rapid prototyping, switch to SGD+Momentum for final training when generalization is critical.

### Theory ⇄ Experiment Validation

See `hypothesis_matrix.md` for complete mapping:

| Hypothesis | Experiment | Visualization |
|------------|------------|---------------|
| Momentum reduces zig-zag | SGD vs SGDM on Rosenbrock | `sgdm_trajectory_series_*.png` |
| Adam accelerates early | MNIST AdamW vs SGD-Momentum | `*_gen_gap.png` |
| Sharp vs flat minima | Loss landscape around trained weights | `loss_landscape_*.png` |
| Layer-wise scaling | Per-layer gradient norms | `*_layer_grads.png` |

## 🛠️ Troubleshooting

**Issue:** Overflow error on Rosenbrock with high momentum
- **Solution:** Reduce learning rate (try 0.001 instead of 0.01)

**Issue:** CUDA out of memory
- **Solution:** Reduce `batch_size` in config or use CPU (default)

**Issue:** Missing eigenvalue columns in old CSV files
- **Solution:** Re-run `python run_experiment.py` to regenerate with new format

## 📚 Citation

If you use this codebase in your research, please cite:

```
@software{gdsearch2025,
  title={GDSearch: Optimizer Dynamics Research Platform},
  author={Le Tran Minh Phuc},
  year={2025},
  url={[https://github.com/Ynhi0/GDSearch]}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Basin-of-attraction maps
- Noisy gradient experiments
- Additional test functions (Beale, Himmelblau, etc.)
- More NN architectures (ResNet, Transformer)

## 📞 Contact

For questions or issues, please open a GitHub issue or contact mphuc666@gmail.com

---

**Last Updated:** November 3, 2025
