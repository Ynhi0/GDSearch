# GDSearch - Optimizer Dynamics Research Platform

A comprehensive Python framework for comparing gradient descent algorithms on 2D test functions and neural networks (MNIST/CIFAR-10/IMDB). Features systematic hyperparameter tuning, convergence analysis, curvature tracking, loss landscape visualization, **multi-seed experiments**, **statistical analysis**, and **NLP support**.

##  Features

### Core Capabilities
-  **4 Optimization Algorithms:** SGD, SGD+Momentum, RMSProp, Adam/AdamW
-  **7 Test Functions:** Rosenbrock, Ill-Conditioned Quadratic, Saddle Point, Rastrigin, Ackley, Sphere, Schwefel
-  **High-Dimensional Benchmarks:** Rastrigin, Ackley, Sphere, Schwefel (N-dimensional, tested up to 100D)
-  **Neural Networks:** SimpleMLP (MNIST), SimpleCNN/ConvNet (CIFAR-10), **ResNet-18** (CIFAR-10), NLP models (IMDB)
-  **Deep Architectures:** ResNet-18 (18 layers, 11M parameters, residual connections)
-  **NLP Models:** SimpleRNN, SimpleLSTM, BiLSTM, TextCNN (Kim 2014)
-  **Systematic Hyperparameter Tuning:** Two-stage pipeline (LR sweep → parameter sweep) + Optuna integration
-  **Learning Rate Schedulers:** Step, Cosine, Exponential, Warmup, OneCycle, and more
-  **Convergence Detection:** Dual conditions (grad norm threshold OR loss delta)
-  **Advanced Analysis:**
  - Hessian eigenvalue tracking (λ_min, λ_max, condition number)
  - Loss landscape 1D/2D visualization
  - Per-layer gradient norms
  - Curvature analysis (trajectory turning angles)
  - Generalization gap monitoring

###  Scientific Rigor
-  **Multi-Seed Experiments:** Run experiments with multiple random seeds for statistical reliability
-  **Statistical Analysis:** T-tests, effect sizes (Cohen's d), 95% confidence intervals
-  **Power Analysis:** Statistical power calculation and sample size determination
-  **Multiple Comparison Corrections:** Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (FDR)
-  **Normality Testing:** Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov
-  **Non-parametric Tests:** Mann-Whitney U, Wilcoxon signed-rank (for non-normal data)
-  **Auto-Test Selection:** Automatically choose appropriate test based on normality
-  **Interactive Visualizations:** Plotly-based 2D/3D plots, animations, loss landscapes
-  **Error Bar Visualization:** Plots with mean ± std bands
-  **Unit Tests:** 177 tests verifying gradients, optimizers, schedulers, NLP, ResNet, high-dim functions, statistics, and visualizations (pytest)
-  **Input Validation:** Comprehensive error checking and input sanitization
-  **Ablation Studies:** Component-wise isolation to quantify contributions
-  **Baseline Comparisons:** Compare custom implementations with PyTorch built-ins
-  **GPU Validation:** Kaggle experiments for large-scale training (ResNet-18: 85.51% on CIFAR-10)

##  Project Structure

```
GDSearch/
 src/                        #  All source code (organized!)
    core/                   # Core implementations
       optimizers.py           # SGD, Adam, RMSProp implementations (2D + ND)
       test_functions.py       # 2D test functions with analytic derivatives
       models.py               # PyTorch NN models (MLP, CNN, ConvNet, ResNet-18)
       nlp_models.py           #  NLP models (RNN, LSTM, BiLSTM, TextCNN)
       nlp_data_utils.py       #  IMDB dataset loading & vocabulary
       pytorch_optimizers.py   #  PyTorch wrappers for custom optimizers
       data_utils.py           # MNIST/CIFAR-10 loaders
       lr_schedulers.py        #  Learning rate scheduling (9 schedulers)
       optuna_tuner.py         #  Optuna hyperparameter optimization
       validation.py           # Input validation & error handling
   
    experiments/            # Experiment runners
       run_experiment.py       # 2D experiments with Hessian tracking
       run_nn_experiment.py    # NN training with convergence detection
       run_multi_seed.py       # Multi-seed experiment framework
       run_full_analysis.py    # Complete pipeline: experiments → stats → plots
   
    analysis/               # Statistical analysis
       statistical_analysis.py # T-tests, effect sizes, confidence intervals
       sensitivity_analysis.py # Hyperparameter sensitivity
       ablation_study.py       # Component-wise ablation
       baseline_comparison.py  # Compare with PyTorch optimizers
   
    visualization/          # Plotting utilities
        plot_results.py         # Comprehensive plotting (with error bars!)
        plot_eigenvalues.py     # Hessian eigenvalue visualization
        loss_landscape.py       # Loss surface probing

 tests/                      # Unit tests (123 tests, 100% passing)
    test_gradients.py       # Numerical gradient verification
    test_optimizers.py      # Optimizer correctness tests
    test_lr_schedulers.py   #  LR scheduler tests
    test_optuna_tuner.py    #  Optuna integration tests
    test_nlp.py             #  NLP models & data tests
    test_resnet.py          #  ResNet-18 architecture tests
    test_highdim_functions.py  #  High-dimensional function tests

 configs/                    # Experiment configurations
    nn_tuning.json          # MNIST hyperparameter sweeps
    cifar10_tuning.json     # CIFAR-10 configurations

 scripts/                    # Utility scripts
    run_all.py              # Complete reproducibility pipeline
    tune_nn.py              # Two-stage hyperparameter tuning
    demo_imdb_training.py   #  IMDB sentiment analysis demo
    demo_highdim_optimization.py  #  High-dimensional function optimization
    generate_summaries.py   # Quantitative & qualitative tables

 docs/                       #  All documentation (consolidated!)
    INDEX.md                # Documentation navigation hub
    LIMITATIONS.md          # Known limitations & assumptions
    MULTISEED_GUIDE.md      # Guide for multi-seed experiments
    IMPROVEMENT_PROGRESS.md # Progress tracking
    CRITICAL_VALIDATION_REPORT.md  # Scientific validation
    REPORT.md               # Synthesis report with ablation study
    PHASE11_NLP_SUMMARY.md  # NLP implementation summary
    PHASE12_RESNET_SUMMARY.md  #  ResNet-18 deep network summary
    hypothesis_matrix.md    # Theory ⇄ Experiment mapping

 kaggle/                     #  Kaggle GPU experiments
    QUICKSTART.md           # How to run experiments on Kaggle
    INSTRUCTIONS.md         # Detailed step-by-step guide
    resnet18_cifar10.py     # ResNet-18 training script
    RESULTS_resnet18.md     #  Kaggle experiment results (85.51% accuracy)
    verify_local.py         # Local verification script

 results/                    # CSV outputs (experiments, summaries)
 plots/                      # All visualizations (PNG)
 data/                       # Dataset utilities

 pyproject.toml              #  Modern Python project configuration
 requirements.txt            # Dependencies
 README.md                   # This file
```

##  Quick Start

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

# Expected: 35 tests passed 
```

### Running Experiments

#### Option 1: Multi-Seed Statistical Analysis (Recommended) 
```bash
# Full pipeline: experiments → aggregation → stats → plots
python src/experiments/run_full_analysis.py \
    --config configs/nn_tuning.json \
    --seeds 1,2,3,4,5 \
    --compare AdamW-SGDMomentum,Adam-RMSProp

# Quick test with 3 seeds
python src/experiments/run_full_analysis.py --seeds 1,2,3
```

**Output:**
- Multi-seed results with mean ± std
- Statistical comparisons (t-tests, p-values, effect sizes)
- Plots with error bars
- Aggregated JSON summary

#### Option 2: Ablation Study (Component Analysis) 
```bash
# Test each optimizer component in isolation
python src/analysis/ablation_study.py

# Components tested:
#   1. SGD baseline (no momentum, no adaptive LR)
#   2. SGD + Momentum
#   3. RMSProp (adaptive LR only)
#   4. Adam (full)
#   5. Adam with L2 regularization
#   6. AdamW (decoupled weight decay)
```

#### Option 3: Baseline Comparison 
```bash
# Compare custom implementations with PyTorch built-ins
python src/analysis/baseline_comparison.py

# Compares:
#   - Custom Adam vs torch.optim.Adam
#   - Custom AdamW vs torch.optim.AdamW
#   - Custom SGD+Momentum vs torch.optim.SGD
#   - Custom RMSProp vs torch.optim.RMSProp
```

#### Option 4: Traditional Single-Seed Pipeline
```bash
# Run everything: 2D + NN tuning + summaries + plots
python scripts/run_all.py

# Skip specific phases
python scripts/run_all.py --skip-2d              # Skip 2D experiments
python scripts/run_all.py --skip-tuning          # Skip NN hyperparameter tuning
python scripts/run_all.py --summaries-only       # Only regenerate summaries

# Quick mode (reduced iterations)
python scripts/run_all.py --quick
```

#### Option 5: Step-by-Step (Learning Mode)

```bash
# 1. Run 2D test function experiments (with Hessian eigenvalue tracking)
python src/experiments/run_experiment.py

# 2. Run neural network hyperparameter tuning
python scripts/tune_nn.py

# 3. Generate loss landscape visualizations
python src/visualization/loss_landscape.py

# 4. Create summary tables and plots
python scripts/generate_summaries.py

# 5. Visualize Hessian eigenvalues (optional)
python src/visualization/plot_eigenvalues.py
```

#### Option 6: Quick Demo
```bash
# Run short MNIST demo (2 epochs)
python scripts/nn_workflow.py
```

#### Option 7: NLP Experiments (NEW! )
```bash
# Train sentiment classifier on IMDB dataset
python scripts/demo_imdb_training.py \
    --model lstm \
    --optimizer adam \
    --epochs 5 \
    --train-size 5000 \
    --test-size 1000

# Available models: rnn, lstm, bilstm, textcnn
# Available optimizers: sgd, sgd_momentum, adam, rmsprop

# Quick test with small dataset
python scripts/demo_imdb_training.py \
    --epochs 2 \
    --train-size 1000 \
    --test-size 200
```

**NLP Models:**
- **SimpleRNN**: Vanilla recurrent network for baseline
- **SimpleLSTM**: Long Short-Term Memory with forget gates
- **BiLSTM**: Bidirectional LSTM for context from both directions
- **TextCNN**: Kim 2014 architecture with multiple filter sizes

**Custom Optimizer Integration:**
- All custom optimizers now support arbitrary-dimensional parameters
- Backward compatible with 2D test functions
- PyTorch-compatible wrappers for seamless neural network training

##  Key Outputs

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

##  Understanding the Outputs

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

##  Advanced Usage

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
# In src/core/test_functions.py
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
# In src/core/optimizers.py
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

##  Results & Insights

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

##  Troubleshooting

**Issue:** Overflow error on Rosenbrock with high momentum
- **Solution:** Reduce learning rate (try 0.001 instead of 0.01)

**Issue:** CUDA out of memory
- **Solution:** Reduce `batch_size` in config or use CPU (default)

**Issue:** Missing eigenvalue columns in old CSV files
- **Solution:** Re-run `python run_experiment.py` to regenerate with new format

##  Citation

If you use this codebase in your research, please cite:

```
@software{gdsearch2025,
  title={GDSearch: Optimizer Dynamics Research Platform},
  author={Le Tran Minh Phuc},
  year={2025},
  url={https://github.com/Ynhi0/GDSearch}
}
```

##  License

MIT License - see LICENSE file for details.

##  Contributing

Contributions welcome! Areas for improvement:
- Basin-of-attraction maps
- Noisy gradient experiments
- Additional test functions (Beale, Himmelblau, etc.)
- More NN architectures (ResNet, Transformer)

##  Contact

For questions or issues, please open a GitHub issue or contact mphuc666@gmail.com

---

**Last Updated:** November 3, 2025