# Dataset Provenance and Version Information

## Overview
This document provides exact dataset sources, versions, and provenance information for all experiments in the GDSearch project. This ensures full reproducibility and scientific rigor as required by the research proposal.

## Datasets Used

### 1. MNIST
- **Source**: torchvision.datasets.MNIST
- **Official URL**: http://yann.lecun.com/exdb/mnist/
- **Version**: Determined by torchvision version (see environment)
- **License**: Public domain (Yann LeCun)
- **Splits**:
  - Train: 60,000 samples
  - Test: 10,000 samples
- **Image Size**: 28×28 grayscale
- **Classes**: 10 (digits 0-9)
- **Citation**:
  ```
  LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
  Gradient-based learning applied to document recognition.
  Proceedings of the IEEE, 86(11), 2278-2324.
  ```

### 2. CIFAR-10
- **Source**: torchvision.datasets.CIFAR10
- **Official URL**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Version**: Determined by torchvision version
- **License**: MIT-like (Krizhevsky & Hinton)
- **Splits**:
  - Train: 50,000 samples
  - Test: 10,000 samples
- **Image Size**: 32×32 RGB
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Citation**:
  ```
  Krizhevsky, A., & Hinton, G. (2009).
  Learning multiple layers of features from tiny images.
  Technical Report, University of Toronto.
  ```

### 3. CIFAR-100
- **Source**: torchvision.datasets.CIFAR100
- **Official URL**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Version**: Determined by torchvision version
- **License**: MIT-like (Krizhevsky & Hinton)
- **Splits**:
  - Train: 50,000 samples
  - Test: 10,000 samples
- **Image Size**: 32×32 RGB
- **Classes**: 100 (20 superclasses, each with 5 subclasses)
- **Normalization**: Mean=(0.5071, 0.4867, 0.4408), Std=(0.2675, 0.2565, 0.2761)
- **Usage**: For testing generalization across larger label spaces

### 4. FashionMNIST
- **Source**: torchvision.datasets.FashionMNIST
- **Official URL**: https://github.com/zalandoresearch/fashion-mnist
- **Version**: Determined by torchvision version
- **License**: MIT
- **Splits**:
  - Train: 60,000 samples
  - Test: 10,000 samples
- **Image Size**: 28×28 grayscale
- **Classes**: 10 (clothing items)
- **Citation**:
  ```
  Xiao, H., Rasul, K., & Vollgraf, R. (2017).
  Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
  arXiv:1708.07747
  ```

### 5. MedMNIST (Medical Imaging)
- **Source**: medmnist Python package
- **Official URL**: https://medmnist.com/
- **Version**: Check with `import medmnist; print(medmnist.__version__)`
- **License**: Apache 2.0
- **Available Datasets**:
  - PathMNIST: Colorectal cancer histology (9 classes, 107,180 images)
  - ChestMNIST: Chest X-rays (14 classes, 112,120 images)
  - DermaMNIST: Dermatoscopy images (7 classes, 10,015 images)
  - OCTMNIST: Retinal OCT images (4 classes, 109,309 images)
  - PneumoniaMNIST: Pediatric chest X-rays (2 classes, 5,856 images)
  - RetinaMNIST: Fundus images (5 classes, 1,600 images)
  - BreastMNIST: Breast ultrasound (2 classes, 780 images)
  - BloodMNIST: Blood cell microscopy (8 classes, 17,092 images)
  - TissueMNIST: Kidney cortex microscopy (8 classes, 236,386 images)
  - OrganAMNIST/OrganCMNIST/OrganSMNIST: Abdominal CT organs (11 classes)
- **Image Size**: 28×28 (grayscale or RGB depending on dataset)
- **Standardization**: Pre-processed to 28×28 for consistency
- **Usage**: **REQUIRED for medical experiments** (not synthetic data)
- **Installation**: `pip install medmnist`
- **Citation**:
  ```
  Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., ... & Ni, B. (2023).
  MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification.
  Scientific Data, 10(1), 41.
  arXiv:2110.14795
  ```

### 6. IMDB (NLP Sentiment Analysis)
- **Source**: HuggingFace datasets library
- **Dataset ID**: `imdb`
- **Official URL**: https://ai.stanford.edu/~amaas/data/sentiment/
- **Version**: Check with `import datasets; print(datasets.__version__)`
- **License**: Custom (see dataset page)
- **Splits**:
  - Train: 25,000 movie reviews
  - Test: 25,000 movie reviews
  - Unsupervised: 50,000 unlabeled reviews (not used)
- **Classes**: 2 (positive, negative sentiment)
- **Note**: May have compatibility issues with Python 3.13. Use Python 3.10 or 3.11 for best results.
- **Citation**:
  ```
  Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
  Learning word vectors for sentiment analysis.
  In Proceedings of the 49th annual meeting of the ACL (pp. 142-150).
  ```

### 7. 2D Test Functions (Mathematical)
- **Source**: src/core/test_functions.py (custom implementation)
- **Functions**:
  - Ackley: Highly multimodal function with many local minima
  - Rosenbrock: Classic non-convex optimization benchmark ("banana function")
  - Rastrigin: Highly multimodal with regularly spaced local minima
  - Sphere: Simple convex quadratic function (baseline)
  - Beale: 2D non-convex function
  - Himmelblau: 4 identical local minima
- **Purpose**: Testing convergence behavior on known mathematical landscapes
- **Usage**: Low-dimensional visualization and theoretical validation

## Dataset Provenance Tracking

All experiments automatically log dataset metadata to MLflow using `src/core/dataset_provenance.py`:

### Logged Metadata
- `dataset_name`: Official name
- `dataset_version`: Library version (e.g., `torchvision_0.15.0`, `medmnist_2.2.3`)
- `dataset_source`: Source library (torchvision, medmnist, huggingface_datasets, etc.)
- `split`: train/val/test
- `num_samples`: Number of samples in split
- `data_seed`: Random seed used for data loading and splitting
- `official_url`: Official dataset homepage
- `citation`: Academic citation
- `timestamp`: ISO timestamp of experiment
- `data_root`: Path to data directory
- `data_root_checksum`: MD5 checksum of data files (for verification)

### Experiment Manifests

Each experiment generates a JSON manifest file with complete provenance:

```json
{
  "experiment_name": "MNIST_SGD_Adam_AdamW",
  "created_at": "2025-12-20T10:30:00.000000",
  "config": {
    "dataset": "MNIST",
    "model": "SimpleMLP",
    "optimizers": ["SGD", "Adam", "AdamW"],
    "seeds": [42, 123, 456],
    "epochs": 10,
    "lr": 0.001
  },
  "dataset_provenance": {
    "dataset_name": "MNIST",
    "dataset_version": "torchvision_0.15.2",
    "data_source": "torchvision",
    "official_url": "http://yann.lecun.com/exdb/mnist/",
    "num_samples": 60000,
    "split": "train"
  },
  "environment": {
    "torch_version": "2.0.1",
    "torchvision_version": "0.15.2",
    "medmnist_version": "2.2.3",
    "datasets_version": "2.14.5"
  }
}
```

## Environment Setup

### Required Packages
```bash
pip install torch torchvision torchaudio  # PyTorch ecosystem
pip install medmnist                       # Medical imaging datasets
pip install datasets transformers          # HuggingFace NLP datasets
pip install numpy pandas matplotlib scipy  # Scientific computing
pip install scikit-learn optuna mlflow     # ML tools
```

### Version Verification
Run this script to verify all datasets are accessible:

```bash
python download_datasets.py
```

For strict MedMNIST requirement (publication mode):
```python
from download_datasets import download_medmnist
download_medmnist('pathmnist', strict=True)  # Raises error if unavailable
```

## Reproducibility Checklist

For each experiment, ensure:
- [ ] Dataset version is logged to MLflow
- [ ] Random seed is set and logged
- [ ] Data loader seed is set (for worker determinism)
- [ ] Normalization parameters are documented
- [ ] Train/val/test splits are reproducible
- [ ] Experiment manifest is generated
- [ ] Results include dataset checksums

## Data Augmentation

### MNIST
- No augmentation (standard practice)
- Normalization: mean=0.1307, std=0.3081

### CIFAR-10/100
- **Train**: RandomCrop(32, padding=4), RandomHorizontalFlip
- **Test**: No augmentation
- **Normalization**:
  - CIFAR-10: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
  - CIFAR-100: mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)

### MedMNIST
- Follow dataset-specific preprocessing (see MedMNIST documentation)
- Default: Resize to 28×28, normalize to [0,1]

## Notes

### Synthetic vs. Real Data
- **Synthetic medical data**: Used for CI/debugging ONLY
- **MedMNIST**: REQUIRED for publication claims
- **Warning**: Experiments using synthetic data are marked with `dataset_source='synthetic'` and include a warning in provenance

### Clinical Datasets
For claims about clinical generalization, consider:
- BraTS (brain tumor segmentation) - requires registration
- LIDC-IDRI (lung CT) - requires DUA
- Kaggle medical datasets (e.g., chest X-ray pneumonia) - variable licensing

### Multi-seed Experiments
All experiments use multiple seeds (minimum 3) for statistical validation:
- Default seeds: 42, 123, 456, 789, 2024
- Results reported as mean ± std
- T-tests performed for optimizer comparisons

## Contact

For dataset issues or questions:
- MedMNIST: https://medmnist.com/
- Torchvision: https://pytorch.org/vision/
- HuggingFace: https://huggingface.co/datasets

## Last Updated
December 20, 2025
