# Optional Dependencies Guide

This document describes optional dependencies for GDSearch and their installation procedures.

## Overview

GDSearch has a minimal core that works with just the requirements in `requirements.txt`. Additional features require optional dependencies that can be installed on demand.

## Optional Feature Packages

### Medical Image Segmentation

**Packages:** `medmnist`, `MONAI`

**Installation:**
```bash
pip install medmnist
pip install 'monai[all]'
```

**When Required:**
- Running medical segmentation experiments (`--experiments medical`)
- Using PathMNIST or other MedMNIST datasets
- Training U-Net models for medical imaging

**Graceful Degradation:**
- If not installed, medical experiments are automatically skipped
- Tests for medical features are skipped (`@pytest.mark.skipif`)
- No errors are raised; experiments continue with other datasets

**Usage Example:**
```bash
# With medical experiments
python run_all_kaggle.py --experiments medical --seeds 42,123,456

# Without medmnist installed, use --require-medmnist to fail fast
python run_all_kaggle.py --experiments medical --require-medmnist
```

---

### NLP / Transformers

**Packages:** `transformers`, `datasets`

**Installation:**
```bash
pip install transformers datasets
```

**When Required:**
- Running NLP experiments (`--experiments nlp`)
- BERT/DistilBERT sentiment analysis on IMDB
- Tokenization and pre-trained transformer models

**Graceful Degradation:**
- NLP experiments are skipped if transformers not available
- Falls back to simpler NLP tasks if datasets unavailable

**Usage Example:**
```bash
# Run NLP sentiment analysis with BERT
python run_all_kaggle.py --experiments nlp --seeds 42,123
```

---

### GPU Monitoring

**Package:** `GPUtil`

**Installation:**
```bash
pip install GPUtil
```

**When Required:**
- GPU memory monitoring during training
- Automatic batch size tuning (`--adaptive-batch`)
- Memory-aware checkpointing

**Graceful Degradation:**
- GPU monitoring is disabled if GPUtil not available
- Warnings logged but training continues normally

---

### Development Tools (Already in requirements-dev.txt)

**Packages:** `pytest`, `black`, `ruff`, `mypy`, `pre-commit`

**Installation:**
```bash
pip install -r requirements-dev.txt
```

**When Required:**
- Running tests (`pytest tests/`)
- Code formatting and linting
- Type checking
- Git pre-commit hooks

---

## Checking Installed Optional Dependencies

```bash
python -c "import medmnist; print('medmnist:', medmnist.__version__)"
python -c "import monai; print('MONAI:', monai.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import GPUtil; print('GPUtil: installed')"
```

---

## Full Installation (All Features)

To install all optional dependencies at once:

```bash
pip install -r requirements.txt
pip install medmnist 'monai[all]' transformers datasets GPUtil
pip install -r requirements-dev.txt  # For development
```

---

## Kaggle Environments

### Pre-installed in Kaggle:
- `transformers`, `datasets` ✅
- `torch`, `torchvision` ✅
- Most common ML packages ✅

### NOT pre-installed in Kaggle:
- `medmnist` ❌ (install in notebook: `!pip install medmnist`)
- `monai` ❌ (install in notebook: `!pip install 'monai[all]'`)
- `GPUtil` ❌ (usually not needed; Kaggle provides GPU monitoring)

**Recommended Kaggle Setup:**
```python
# At top of Kaggle notebook
!pip install medmnist 'monai[all]' -q

import sys
sys.path.append('/kaggle/working')  # Add working directory to path
```

---

## Troubleshooting

### MONAI Installation Issues

If you encounter errors installing MONAI, try installing with minimal dependencies:
```bash
pip install monai  # Core only, no optional features
```

Or install specific extras:
```bash
pip install 'monai[nibabel,tqdm]'  # Specific features only
```

### Transformers Model Download

If you're behind a firewall or have network issues, pre-download models:
```python
from transformers import AutoModel, AutoTokenizer

# Download once (cached for future use)
AutoModel.from_pretrained('bert-base-uncased')
AutoTokenizer.from_pretrained('bert-base-uncased')
```

### Python Version Compatibility

- **Python 3.8-3.12:** All packages supported ✅
- **Python 3.13+:** MONAI may have compatibility issues ⚠️

Check compatibility before upgrading Python:
```bash
python --version
pip check  # Verify no broken dependencies
```

---

## Related Documentation

- [README.md](../README.md): Quick start and basic usage
- [docs/guides/EXPERIMENT_EXECUTION_GUIDE.md](guides/EXPERIMENT_EXECUTION_GUIDE.md): Running experiments
- [requirements.txt](../requirements.txt): Core dependencies
- [requirements-dev.txt](../requirements-dev.txt): Development dependencies

---

**Last Updated:** February 2, 2026  
**Maintainer:** GDSearch Team
