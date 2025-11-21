# Kaggle Benchmark Experiments

This folder contains ready-to-run benchmark experiments for Kaggle notebooks with GPU acceleration. All experiments are designed for reproducible research with multi-seed statistical validation.

##  Structure

```
kaggle/
 README.md                    # This file (comprehensive guide)
 analysis_visualization.ipynb # Publication-quality plots and analysis
 requirements_kaggle.txt      # Kaggle-specific dependencies

 mnist_benchmark/            # MNIST experiments (7 optimizers × N seeds)
     run_mnist.py            # Main experiment script
     run_mnist.ipynb         # Jupyter notebook version
     requirements.txt        # Additional dependencies

 cifar10_benchmark/          # CIFAR-10 experiments (ResNet-18)
     run_cifar10.py          # ResNet-18 training script
     run_cifar10.ipynb       # Jupyter notebook version

 nlp_benchmark/              # IMDB sentiment analysis
     run_nlp.py              # NLP model training
     run_nlp.ipynb           # Jupyter notebook version

 medical_benchmark/          # Medical image segmentation
     run_seg.py              # Segmentation experiments
     run_seg.ipynb           # Jupyter notebook version

 notebooks/                  # Legacy notebooks
 resnet18_experiment/        # ResNet-18 experiment artifacts
```

##  Available Benchmarks

### 1. MNIST Benchmark (`mnist_benchmark/`)
- **Model**: SimpleMLP (256 → 128 → 10)
- **Dataset**: MNIST (60K train, 10K test)
- **Optimizers**: 7 optimizers × N seeds
  - SGD (lr=0.01)
  - SGD_Momentum (lr=0.05, momentum=0.9)
  - Adam (lr=0.001)
  - AdamW (lr=0.001, weight_decay=1e-4)
  - AMSGrad (lr=0.001, amsgrad=True)
  - **SAM_SGD** (lr=0.01, rho=0.05) - Sharpness-Aware Minimization
  - **SAM_Adam** (lr=0.001, rho=0.05) - Sharpness-Aware Minimization
- **Output**: Per-run CSVs + statistical comparison CSV
- **Statistical Tests**: Paired t-tests + Holm-Bonferroni correction

### 2. CIFAR-10 Benchmark (`cifar10_benchmark/`)
- **Model**: ResNet-18 (11M parameters)
- **Dataset**: CIFAR-10 (50K train, 10K test)
- **Training**: Full training pipeline with data augmentation
- **Expected Accuracy**: ~85-87% on test set

### 3. NLP Benchmark (`nlp_benchmark/`)
- **Models**: BiLSTM, TextCNN (Kim 2014)
- **Dataset**: IMDB sentiment analysis
- **Features**: Pre-trained embeddings, attention mechanisms

### 4. Medical Benchmark (`medical_benchmark/`)
- **Task**: Medical image segmentation
- **Models**: U-Net variants
- **Dataset**: Medical imaging datasets

##  How to Run Experiments

### Option 1: Upload Python Script (Recommended)
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Copy contents of desired `run_*.py` script
4. Paste into Kaggle notebook
5. Enable GPU: Settings → Accelerator → GPU T4 x2
6. Run All

### Option 2: Upload Notebook
1. Upload the corresponding `run_*.ipynb` notebook
2. Enable GPU acceleration
3. Run cells sequentially

### Option 3: Run All Benchmarks
```bash
# From project root
python scripts/run_final_benchmarks.py --seeds 1,2,3,4,5
```

##  Kaggle Settings

**Recommended Configuration:**
- **Accelerator**: GPU T4 x2 (or P100)
- **Internet**: On (for dataset downloads)
- **Environment**: Latest (Python 3.10+, PyTorch 2.0+)
- **Persistence**: Files only (for output saving)

##  Output and Analysis

### Generated Files
- `*_benchmark.csv`: Per-run results (train/test loss, accuracy, timing)
- `statistical_comparison.csv`: Statistical analysis across optimizers
- `analysis_visualization.ipynb`: Publication-ready plots

### Analysis Scripts
```bash
# Flatness analysis (SAM vs traditional optimizers)
python analyze_flatness.py --results_dir kaggle/working/results/

# Loss landscape visualization
python visualize_flatness_comparison.py --adam_model adam.pt --sam_model sam.pt
```

##  SAM Optimizer Details

The SAM (Sharpness-Aware Minimization) optimizer is included in MNIST benchmarks:

- **Algorithm**: Minimizes loss + sharpness (worst-case loss in neighborhood)
- **Implementation**: Dual forward/backward pass per step (2x computational cost)
- **Benefit**: Finds flatter minima with better generalization
- **Trade-off**: Slower training but improved test performance

##  Troubleshooting

**Common Issues:**
- **CUDA out of memory**: Reduce batch size or use CPU
- **Dataset download fails**: Check internet connection
- **Import errors**: Install requirements: `pip install -r requirements_kaggle.txt`

**Performance Tips:**
- Use GPU T4 x2 for fastest training
- MNIST benchmarks complete in ~10-15 minutes
- CIFAR-10 ResNet-18 takes ~30-45 minutes
   - Confirm custom optimizers work
   - GPU utilization (if visible)

3. **Screenshots** (optional):
   - Training progress
   - Final results

##  Current Experiments

### 1. ResNet-18 on CIFAR-10 (Phase 12)
**Purpose**: Verify custom optimizers work with deep networks (skip connections)

**Files**: 
- `resnet18_cifar10.py`
- `notebooks/resnet18_demo.ipynb`

**Expected Runtime**: ~5-10 minutes (5 epochs on GPU)

**What we're testing**:
-  Custom Adam works with 11M parameter model
-  Gradient flow through residual connections
-  Deep network (18 layers) compatibility

---

##  Notes

- All code is self-contained (no external imports needed)
- CIFAR-10 dataset downloads automatically
- Results can be copy-pasted back to continue development
