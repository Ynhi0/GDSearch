# GDSearch - Optimizer Dynamics Research Platform

A comprehensive Python framework for comparing gradient descent algorithms on 2D test functions and neural networks (MNIST/CIFAR-10/IMDB). Features systematic hyperparameter tuning, convergence analysis, curvature tracking, loss landscape visualization, **multi-seed experiments**, **statistical analysis**, and **NLP support**.

### Core Capabilities

- **12 Optimization Algorithms:** SGD, SGDMomentum, SGDNesterov, RMSProp, Adam, AdamW, AMSGrad, **SAM** (Sharpness-Aware Minimization), **Lookahead**, AdaBound, RAdam, and LAMB
- **7 Test Functions:** Rosenbrock, Ill-Conditioned Quadratic, Saddle Point, Ackley2D, Rastrigin, Ackley, Sphere, Schwefel
- **High-Dimensional Benchmarks:** Rastrigin, Ackley, Sphere, Schwefel (N-dimensional, tested up to 100D)
- **Neural Networks:** SimpleMLP (MNIST), SimpleCNN/ConvNet (CIFAR-10), **ResNet-18** (CIFAR-10), NLP models (IMDB)
- **Deep Architectures:** ResNet-18 (18 layers, 11M parameters, residual connections)
- **NLP Models:** SimpleRNN, SimpleLSTM, BiLSTM, TextCNN (Kim 2014)
- **Systematic Hyperparameter Tuning:** Two-stage pipeline (LR sweep → parameter sweep) + Optuna integration
- **Learning Rate Schedulers:** Constant, Step, MultiStep, Exponential, Cosine, Warmup, Polynomial, OneCycle (9 schedulers)
- **Convergence Detection:** Dual conditions (grad norm threshold OR loss delta)
- **Automatic Visualization:** Every experiment generates both static (PNG) and interactive (HTML) plots immediately after completion
- **Advanced Analysis:**
- **Hessian eigenvalue tracking** (λ_min, λ_max, condition number) with **proper deflation**
- Loss landscape 1D/2D visualization
- **Flatness Analysis:** Training stability, generalization gap, loss smoothness metrics
- **SAM Minima Visualization:** Contour plots comparing Adam vs SAM minima flatness
- Per-layer gradient norms
- Curvature analysis (trajectory turning angles)
- Generalization gap monitoring

### Scientific Rigor

- **Multi-Seed Experiments:** Run experiments with multiple random seeds for statistical reliability
- **Statistical Analysis:** T-tests, effect sizes (Cohen's d), 95% confidence intervals
- **Power Analysis:** Statistical power calculation and sample size determination
- **Multiple Comparison Corrections:** Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (FDR)
- **Normality Testing:** Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov
- **Non-parametric Tests:** Mann-Whitney U, Wilcoxon signed-rank (for non-normal data)
- **Auto-Test Selection:** Automatically choose appropriate test based on normality
- **Interactive Visualizations:** Plotly-based 2D/3D plots, animations, loss landscapes
- **Error Bar Visualization:** Plots with mean ± std bands
- **Adaptive Overfitting Prevention:** Enforced train/val/test separation (BLOCKER-1 fix)
- **Checkpoint Robustness:** Complete state saving including scheduler/scaler/EMA (BLOCKER-2 fix)
- **Config Validation:** Automated schema checks prevent silent errors (BLOCKER-3 fix)
- **Search Budget Parity:** Automated fairness checks across optimizers (HIGH-2 fix)

### Modern Optimization Techniques

- **SAM (Sharpness-Aware Minimization):** Finds flatter minima for better generalization (ICLR 2021)
- **Lookahead Optimizer:** Meta-optimizer with slow/fast weights for stability (NeurIPS 2019)
- **Flatness Analysis:** Quantitative metrics for minimum quality assessment
- **Computational Cost Analysis:** Wall-clock time metrics (SAM requires 2x forward/backward passes)
- **Unit Tests:** 208+ tests verifying gradients, optimizers, schedulers, NLP, ResNet, checkpoints, tuning safety, and more (pytest)
- **Input Validation:** Comprehensive error checking and input sanitization
- **Ablation Studies:** Component-wise isolation to quantify contributions
- **Baseline Comparisons:** Compare custom implementations with PyTorch built-ins
- **GPU Validation:** Kaggle experiments for large-scale training (ResNet-18: 85.51% on CIFAR-10)
- **CI/CD:** Automated config validation, budget parity checks, tuning safety lint

## Supported Datasets

### Computer Vision Datasets

- **MNIST**: 60,000 training, 10,000 test images (28×28 grayscale handwritten digits)
- **FashionMNIST**: 60,000 training, 10,000 test images (28×28 grayscale fashion items)
- **CIFAR-10**: 50,000 training, 10,000 test images (32×32 RGB, 10 classes)
- **CIFAR-100**: 50,000 training, 10,000 test images (32×32 RGB, 100 classes)

### Natural Language Processing Datasets

- **IMDB Movie Reviews**: 25,000 training, 25,000 test reviews (binary sentiment classification)

### Test Functions (2D)

- **Rosenbrock**: Banana-shaped valley, tests optimizer ability to handle non-convex landscapes
- **Ackley2D**: Multimodal function with many local minima, tests global optimization
- **Rastrigin**: Highly multimodal with regular pattern of local minima
- **Ill-Conditioned Quadratic**: Tests conditioning and convergence speed
- **Saddle Point**: Tests behavior near inflection points
- **Sphere**: Simple convex function for baseline comparisons
- **Schwefel**: Complex multimodal function with deceptive minima

### High-Dimensional Benchmarks

- **Rastrigin (N-D)**: Multimodal function scalable to arbitrary dimensions (tested up to 100D)
- **Ackley (N-D)**: Multimodal function with exponential decay
- **Sphere (N-D)**: Simple convex function in high dimensions
- **Schwefel (N-D)**: Complex deceptive function in high dimensions

## Supported Optimizers

GDSearch implements 12 state-of-the-art optimization algorithms:

### First-Order Methods

1. **SGD**: Vanilla Stochastic Gradient Descent (lr=0.01)
2. **SGDMomentum**: SGD with momentum accumulation (lr=0.01, momentum=0.9)
3. **SGDNesterov**: Nesterov accelerated gradient (lr=0.01, momentum=0.9)

### Adaptive Learning Rate Methods

4. **RMSProp**: Root Mean Square Propagation (lr=0.001, alpha=0.99, eps=1e-8)
5. **Adam**: Adaptive Moment Estimation (lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
6. **AdamW**: Adam with decoupled weight decay (lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01)
7. **AMSGrad**: Adam with max instead of exponential moving average (lr=0.001, beta1=0.9, beta2=0.999)

### Advanced Methods

8. **SAM (Sharpness-Aware Minimization)**: Finds flatter minima for better generalization (lr=0.01, rho=0.05)
9. **Lookahead**: Meta-optimizer with slow/fast weights for stability (lr=0.01, k=5, alpha=0.5)
10. **AdaBound**: Adaptive bound on learning rate (lr=0.001, final_lr=0.1, gamma=1e-3)
11. **RAdam**: Rectified Adam with warmup (lr=0.001, beta1=0.9, beta2=0.999)
12. **LAMB**: Layer-wise Adaptive Moments for Batching (lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01)

All optimizers support both 2D test functions and high-dimensional neural network training with automatic parameter handling and PyTorch compatibility.

## Project Structure

```
GDSearch/
├── src/                        # All source code (organized!)
│   ├── core/                   # Core implementations
│   │   ├── optimizers.py           # SGD, Adam, RMSProp, SAM, Lookahead (2D + ND)
│   │   ├── test_functions.py       # 2D test functions with analytic derivatives
│   │   ├── models.py               # PyTorch NN models (MLP, CNN, ConvNet, ResNet-18)
│   │   ├── nlp_models.py           # NLP models (RNN, LSTM, BiLSTM, TextCNN)
│   │   ├── nlp_data_utils.py       # IMDB dataset loading & vocabulary
│   │   ├── pytorch_optimizers.py   # PyTorch wrappers for custom optimizers
│   │   ├── data_utils.py           # MNIST/CIFAR-10 loaders
│   │   ├── lr_schedulers.py        # Learning rate scheduling (9 schedulers)
│   │   ├── optuna_tuner.py         # Optuna hyperparameter optimization
│   │   ├── validation.py           # Input validation & error handling
│   │   └── optimizer_wrappers.py   # Additional optimizer utilities
│   ├── experiments/            # Experiment runners
│   │   ├── run_experiment.py       # 2D experiments with Hessian tracking
│   │   ├── run_nn_experiment.py    # NN training with convergence detection
│   │   ├── run_multi_seed.py       # Multi-seed experiment framework
│   │   ├── run_full_analysis.py    # Complete pipeline: experiments → stats → plots
│   │   ├── run_cifar10.py          # CIFAR-10 specific experiments
│   │   ├── run_initial_condition_robustness.py  # Robustness analysis
│   │   ├── run_medical_segmentation.py  # Medical imaging experiments
│   │   ├── run_optimizer_ablation.py   # Optimizer ablation studies
│   │   └── run_transformer_nlp.py     # Transformer NLP fine-tuning
│   ├── analysis/               # Statistical analysis
│   │   ├── statistical_analysis.py # T-tests, effect sizes, confidence intervals
│   │   ├── sensitivity_analysis.py # Hyperparameter sensitivity
│   │   ├── ablation_study.py       # Component-wise ablation
│   │   └── baseline_comparison.py  # Compare with PyTorch optimizers
│   ├── runners/                # Specialized experiment runners
│   │   ├── cifar10_runner.py       # CIFAR-10 specific runner
│   │   ├── mnist_runner.py         # MNIST specific runner
│   │   └── __init__.py
│   ├── utils/                  # Utility functions
│   └── visualization/          # Plotting utilities
│       ├── plot_results.py         # Comprehensive plotting (with error bars!)
│       ├── plot_eigenvalues.py     # Hessian eigenvalue visualization
│       ├── loss_landscape.py       # Loss surface probing
│       ├── interactive_plots.py    # Interactive Plotly visualizations
│       ├── create_separate_plots.py # Separate plot generation
│       └── run_loss_landscape.py   # Loss landscape runner
├── tests/                      # Unit tests (200+ tests covering all components)
│   ├── test_gradients.py       # Numerical gradient verification
│   ├── test_optimizers.py      # Optimizer correctness tests
│   ├── test_lr_schedulers.py   # LR scheduler tests
│   ├── test_optuna_tuner.py    # Optuna integration tests
│   ├── test_nlp.py             # NLP models & data tests
│   ├── test_resnet.py          # ResNet-18 architecture tests
│   ├── test_highdim_functions.py  # High-dimensional function tests
│   ├── test_ackley2d.py        # Ackley 2D function tests
│   ├── test_interactive_plots.py # Interactive plotting tests
│   └── test_statistical_enhancements.py # Statistical analysis tests
├── configs/                    # Experiment configurations
│   ├── benchmark_hyperparameters.json
│   ├── cifar10_tuning.json     # CIFAR-10 configurations
│   ├── config_schema.json
│   └── nn_tuning.json          # MNIST hyperparameter sweeps
├── scripts/                    # Utility scripts
│   ├── analyze_lr_finder_efficacy.py
│   ├── check_search_budget_parity.py
│   ├── compute_tradeoffs.py
│   ├── demo_highdim_optimization.py  # High-dimensional function optimization
│   ├── demo_lr_schedulers.py   # LR scheduler demonstration
│   ├── demo_imdb_training.py   # IMDB sentiment analysis demo
│   ├── diagnose_adam.py
│   ├── diagnose_imports.py
│   ├── diagnose_sgd_momentum.py
│   ├── generate_appendix.py
│   ├── generate_cifar10_statistical_report.py
│   ├── generate_experiment_plots.py
│   ├── generate_latex_tables.py
│   ├── generate_statistical_report.py
│   ├── generate_summaries.py
│   ├── optuna_tune_mnist.py
│   ├── quick_validation_test.py
│   ├── run_all.py              # Complete reproducibility pipeline
│   ├── run_experiment_with_analysis.py
│   ├── run_final_benchmarks.py
│   ├── run_mnist_full.py
│   ├── run_nn_ablation.py
│   ├── run_pytest.py
│   ├── so_what_analysis.py
│   ├── train_lstm_imdb.py
│   ├── train_resnet18_cifar10.py
│   ├── tune_nn.py              # Two-stage hyperparameter tuning
│   ├── validate_all_experiments.py
│   ├── validate_all_fixes.py
│   ├── validate_audit_fixes_quick.py
│   ├── validate_configs.py
│   ├── validate_config_schema.py
│   ├── validate_experiment_config.py
│   ├── validate_neurips_fixes.py
│   ├── validate_remediation_fixes_quick.py
│   ├── verify_all_audit_fixes.py
│   ├── archive/                # Archived scripts
│   └── legacy/                 # Legacy scripts
├── docs/                       # Documentation
│   ├── DATASET_PROVENANCE.md
│   ├── PYTHON313_IMDB_ISSUE.md
│   ├── proposal_text.txt
│   └── image/
├── kaggle/                     # Kaggle GPU experiments
│   ├── analysis_visualization.ipynb
│   ├── cifar10_benchmark/      # CIFAR-10 benchmark experiments
│   ├── fashion_mnist_benchmark/ # FashionMNIST benchmark experiments
│   ├── gradient_monitoring.py
│   ├── medical_benchmark/      # Medical imaging benchmarks
│   ├── mlruns/                 # MLflow tracking
│   ├── mnist_benchmark/        # MNIST benchmark experiments
│   ├── nlp_benchmark/          # NLP benchmark experiments
│   ├── nlp_benchmark.py
│   ├── requirements_kaggle.txt
│   ├── resnet18_cifar10.py     # ResNet-18 training script
│   ├── run_benchmark.ipynb
│   ├── validate_dependencies.py
│   ├── verify_local.py
│   ├── visualize_landscape.py
│   └── __pycache__/
├── results/                    # CSV outputs (experiments, summaries)
│   ├── ablation_studies/
│   ├── analysis/
│   ├── checkpoints/
│   ├── experiments/
│   ├── fair_ablation/
│   ├── optuna_results.json
│   ├── reports/
│   ├── test/
│   ├── test2/
│   └── visualizations/
├── data/                       # Dataset utilities
│   ├── cifar-10-batches-py/
│   ├── cifar-100-python/
│   ├── FashionMNIST/
│   └── MNIST/
├── mlruns/                     # MLflow experiment tracking
├── download_datasets.py
├── download_datasets_kaggle.py
├── install_dependencies.py
├── pytest.ini
├── requirements.txt
├── run_all_kaggle.py
├── README.md                   # This file
├── .git/
├── .github/
├── .gitignore
├── .pylintrc
├── .pytest_cache/
├── .qodo/
├── venv/                       # Virtual environment
└── __pycache__/
```

## Quick Start

### Installation

```bash
# Clone or navigate to the project
cd /workspaces/GDSearch

# Install dependencies
pip install -r requirements.txt
```

> **Kaggle users:** Kaggle kernels use modern NumPy 2.x (optimized and performant). Our codebase is NumPy 2.x compatible. If you encounter binary incompatibility errors:
>
> 1. **Best practice (automatic in notebook):** Let the compatibility check cell reinstall Pandas to match NumPy 2.x
> 2. **Manual fix if needed:**
>    ```bash
>    pip install --force-reinstall --no-cache-dir --no-deps pandas
>    pip install pandas  # Reinstall dependencies
>    ```
> 3. **Restart the kernel** after any reinstallation
>
> The notebook automatically handles this in the pre-install compatibility check cell.

**Dependencies:** numpy, pandas, matplotlib, scipy, torch, torchvision, tqdm, pytest, optuna, datasets, plotly

### Running Tests (Verify Installation)

```bash
# Run all tests (gradients + optimizers)
pytest tests/ -v

# Expected: 200+ tests covering all components 
```

### Running Experiments

#### Main Entry Point: `run_all_kaggle.py`

The primary interface for running experiments is `run_all_kaggle.py`, which supports comprehensive benchmarking across all datasets and optimizers.

```bash
# Quick test with 3 seeds (ultra-fast mode)
python run_all_kaggle.py --ultra-quick --seeds 42,123,456

# Full reproducible run with 10 seeds (statistical rigor)
python run_all_kaggle.py --seeds 42,123,456,789,1011,1213,1415,1617,1819,2021

# Run specific experiments only
python run_all_kaggle.py --experiments mnist,cifar10 --quick

# Skip hyperparameter tuning (use defaults)
python run_all_kaggle.py --skip-tuning

# Kaggle T4 GPU optimizations
python run_all_kaggle.py --kaggle-t4 --quick

# Resume from partial results
python run_all_kaggle.py --resume

# Enable advanced features
python run_all_kaggle.py --use-amp --use-ema --label-smoothing 0.1
```

**Available Experiments:**

- `mnist`: MNIST handwritten digit classification
- `cifar10`: CIFAR-10 image classification
- `nlp`: IMDB sentiment analysis (requires transformers, datasets)
- `medical`: Medical image segmentation (MONAI framework)
- `2d`: 2D test function optimization
- `robustness`: Initial condition robustness analysis
- `sam`: SAM sensitivity analysis
- `ablation`: Optimizer component ablation studies
- `resnet`: ResNet-18 deep network training
- `highdim`: High-dimensional optimization
- `all`: Run all experiments (default)

#### Alternative Entry Points

```bash
# Legacy comprehensive pipeline (2D + NN + summaries)
python scripts/run_all.py

# Individual experiment scripts
python scripts/train_resnet18_cifar10.py --epochs 2
python scripts/train_lstm_imdb.py --model lstm --optimizer adam

# Validation and testing
python scripts/quick_validation_test.py  # Fast validation
python scripts/validate_all_experiments.py  # Comprehensive validation
```

## Key Outputs

### Results Directory (`results/`)

**Experiment CSVs:**

- `NN_<model>_<dataset>_<optimizer>_lr<lr>_seed<seed>[_tag].csv`
- Tags: `sweepLR`, `sweepWD`, `sweepMOM`, `final`

**Summary Files:**

- `summary_quantitative.csv`: Final metrics, convergence iters/time
- `summary_qualitative.md`: Smoothness, oscillation ratings
- `statistical_comparisons.csv`: T-tests, p-values, effect sizes
- `multiseed_detailed.csv`: Aggregated multi-seed results

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

### MLflow Tracking (`mlruns/`)

- Automatic experiment tracking with parameters, metrics, and artifacts
- Accessible via `mlflow ui` command

## Visualization Scripts

### `create_separate_plots.py` - Multi-Panel Analysis

Creates 6 high-resolution PNG plots (300 DPI) from multi-seed experiment results:

```bash
python src/visualization/create_separate_plots.py \
    --summary results/optimizer_summary.csv \
    --stats results/statistical_comparisons.csv \
    --detailed results/multiseed_detailed.csv \
    --output plots
```

**Generated Plots:**

1. **`01_final_loss_comparison.png`**: Final loss comparison with error bars
2. **`02_distance_to_optimum.png`**: Distance to global optimum (1,1)
3. **`03_convergence_rate.png`**: Success rate across seeds
4. **`04_loss_distribution_boxplot.png`**: Loss distribution across seeds
5. **`05_training_curves.png`**: Loss curves with confidence bands
6. **`06_statistical_summary.png`**: T-test results and effect sizes

### `plot_eigenvalues.py` - Curvature Analysis

Visualizes Hessian eigenvalue evolution during optimization:

```bash
python src/visualization/plot_eigenvalues.py
```

**Shows:** λ_min, λ_max, condition number over training iterations.

### Report Generation Scripts

```bash
# Generate comprehensive statistical reports
python scripts/generate_statistical_report.py
python scripts/generate_cifar10_statistical_report.py

# Generate LaTeX tables for papers
python scripts/generate_latex_tables.py

# Generate summary tables
python scripts/generate_summaries.py
```

## Understanding the Outputs

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

## Advanced Usage

### Custom Hyperparameter Tuning

Edit `configs/nn_tuning.json` or `configs/cifar10_tuning.json`:

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

Then run: `python scripts/tune_nn.py`

### Validation and Testing Scripts

```bash
# Run comprehensive validation suite
python scripts/validate_all_experiments.py

# Quick validation test
python scripts/quick_validation_test.py

# Check configuration schema
python scripts/validate_config_schema.py

# Verify all fixes are working
python scripts/verify_all_audit_fixes.py
```

### Analysis and Reporting Scripts

```bash
# Compute tradeoffs between optimizers
python scripts/compute_tradeoffs.py

# Analyze LR finder efficacy
python scripts/analyze_lr_finder_efficacy.py

# Check search budget parity
python scripts/check_search_budget_parity.py

# Generate comprehensive reports
python scripts/generate_appendix.py
python scripts/so_what_analysis.py
```

## Results & Insights

### Optimizer Performance Summary

Based on comprehensive benchmarking across MNIST, CIFAR-10, and 2D test functions:

| Optimizer             | MNIST Test Acc | CIFAR-10 Test Acc | Convergence Speed   | Memory Usage | Notes                                |
| --------------------- | -------------- | ----------------- | ------------------- | ------------ | ------------------------------------ |
| **AdamW**       | ~97.5%         | ~85-87%           | Fast (early epochs) | Moderate     | Best overall for deep networks       |
| **SGDMomentum** | ~97.6%         | ~83-85%           | Slower start        | Low          | Excellent generalization             |
| **SAM**         | ~97.2%         | ~86-88%           | Slow (2x cost)      | High         | Flattest minima, best generalization |
| **Adam**        | ~97.3%         | ~84-86%           | Fast                | Moderate     | Good baseline, some overfitting      |
| **RMSProp**     | ~96.8%         | ~82-84%           | Medium              | Moderate     | Stable for RNNs                      |
| **SGD**         | ~95.5%         | ~78-80%           | Variable            | Low          | Simple but requires tuning           |

### Key Findings

- **AdamW** provides the best trade-off between speed and accuracy for deep networks
- **SAM** finds flatter minima but requires 2x computational cost
- **SGDMomentum** shows superior generalization on some datasets
- **Lookahead** provides stability improvements when combined with other optimizers
- All optimizers benefit from proper hyperparameter tuning

### Theory vs Experiment Validation

The codebase validates multiple optimization theory hypotheses:

| Hypothesis                   | Experimental Validation                            | Key Insight                                       |
| ---------------------------- | -------------------------------------------------- | ------------------------------------------------- |
| Momentum reduces oscillation | Rosenbrock trajectories show smoother paths        | Confirmed: β=0.9 eliminates zig-zag              |
| Adam accelerates convergence | MNIST training curves show faster initial progress | Confirmed: 2-3x faster than SGD                   |
| Sharp vs flat minima         | SAM vs Adam loss landscapes                        | SAM finds significantly flatter minima            |
| Layer-wise scaling           | Per-layer gradient norm analysis                   | Different layers require different learning rates |
| Adaptive methods help        | RMSProp vs SGD on ill-conditioned problems         | Confirmed: Better conditioning handling           |

## Troubleshooting

### Common Issues

**Issue:** Import errors or missing dependencies

- **Solution:** Run `python scripts/install_dependencies.py` or `pip install -r requirements.txt`

**Issue:** CUDA out of memory during training

- **Solution:** Reduce batch size, use `--quick` mode, or set smaller model sizes

**Issue:** Unicode encoding errors on Windows

- **Solution:** The codebase includes automatic fallback for Windows console encoding issues

**Issue:** Slow performance or hanging

- **Solution:** Use `--quick` or `--ultra-quick` modes for testing, check GPU memory usage

**Issue:** Missing MLflow UI

- **Solution:** Install MLflow (`pip install mlflow`) and run `mlflow ui` in the project directory

**Issue:** Configuration validation errors

- **Solution:** Run `python scripts/validate_config_schema.py` to check your config files

**Issue:** Experiment resume not working

- **Solution:** Ensure CSV files exist in the results directory from previous runs

### Validation Scripts

```bash
# Run all validation checks
python scripts/validate_all_experiments.py

# Check for import safety issues
python scripts/diagnose_imports.py

# Verify optimizer implementations
python scripts/validate_all_fixes.py

# Test configuration parsing
python scripts/validate_experiment_config.py
```

## Citation

If you use this codebase in your research, please cite:

```
@software{gdsearch2025,
  title={GDSearch: Optimizer Dynamics Research Platform},
  author={Le Tran Minh Phuc},
  year={2025},
  url={https://github.com/Ynhi0/GDSearch}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact mphuc666@gmail.com
