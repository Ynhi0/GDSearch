# Kaggle Experiments

This folder contains ready-to-run code for Kaggle notebooks with GPU acceleration.

## 📁 Structure

```
kaggle/
├── README.md                    # This file
├── resnet18_cifar10.py          # ResNet-18 training script
├── requirements_kaggle.txt      # Kaggle-specific dependencies
└── notebooks/                   # Jupyter notebooks for Kaggle
    └── resnet18_demo.ipynb      # Ready-to-upload notebook
```

## 🚀 How to Use

### Option 1: Upload Python Script
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Copy contents of `resnet18_cifar10.py`
4. Paste into Kaggle notebook
5. Enable GPU: Settings → Accelerator → GPU T4 x2
6. Run All

### Option 2: Upload Notebook
1. Go to https://www.kaggle.com/code
2. Click "Import Notebook"
3. Upload `notebooks/resnet18_demo.ipynb`
4. Enable GPU: Settings → Accelerator → GPU T4 x2
5. Run All

## ⚙️ Settings

**Recommended Kaggle Settings:**
- **Accelerator**: GPU T4 x2 (or P100)
- **Internet**: On (for downloading CIFAR-10)
- **Environment**: Latest (Python 3.10+, PyTorch 2.0+)

## 📊 What to Report Back

After running on Kaggle, please share:

1. **Training Output**:
   - Loss values per epoch
   - Training/test accuracy
   - Total training time

2. **Verification**:
   - Confirm no errors
   - Confirm custom optimizers work
   - GPU utilization (if visible)

3. **Screenshots** (optional):
   - Training progress
   - Final results

## 🎯 Current Experiments

### 1. ResNet-18 on CIFAR-10 (Phase 12)
**Purpose**: Verify custom optimizers work with deep networks (skip connections)

**Files**: 
- `resnet18_cifar10.py`
- `notebooks/resnet18_demo.ipynb`

**Expected Runtime**: ~5-10 minutes (5 epochs on GPU)

**What we're testing**:
- ✅ Custom Adam works with 11M parameter model
- ✅ Gradient flow through residual connections
- ✅ Deep network (18 layers) compatibility

---

## 📝 Notes

- All code is self-contained (no external imports needed)
- CIFAR-10 dataset downloads automatically
- Results can be copy-pasted back to continue development
