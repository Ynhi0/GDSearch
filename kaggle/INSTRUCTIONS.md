#  Kaggle Experiment Instructions

## Current Experiment: ResNet-18 on CIFAR-10

###  Purpose
Verify that custom optimizers work with deep networks (18 layers, 11M parameters, skip connections).

This addresses **Limitation #8** from `docs/LIMITATIONS.md`: Model Architectures - Deep Networks.

---

##  Step-by-Step Instructions

### 1. Open Kaggle
Go to: https://www.kaggle.com/code

### 2. Create New Notebook
- Click **"New Notebook"** button
- Or click **"Import Notebook"** if using .ipynb file

### 3. Configure Settings
**IMPORTANT**: Enable GPU!
- Click **Settings** (gear icon, top right)
- Under **Accelerator**, select: **GPU T4 x2** (or P100)
- Under **Internet**, toggle: **ON** (needed for CIFAR-10 download)
- Click **Save**

### 4. Add Code
Copy the entire contents of `resnet18_cifar10.py` and paste into a code cell.

### 5. Run
- Click **"Run All"** (or Shift+Enter)
- Wait ~5-10 minutes for 5 epochs

### 6. Copy Results
Once complete, copy the **entire output** including:
- Training progress (loss/accuracy per epoch)
- Best test accuracy
- Total training time
- Verification checklist at the end

### 7. Report Back
Paste the output in the project conversation or create a file `kaggle/results_resnet18.txt`

---

##  Expected Output

You should see something like:

```
 Using device: cuda
   GPU: Tesla T4

================================================================================
ResNet-18 on CIFAR-10 with Custom Adam Optimizer
================================================================================

 Loading CIFAR-10 dataset...
 Train samples: 50,000
 Test samples: 10,000
...

Epoch 1/5
--------------------------------------------------------------------------------
Training: 100%|| 391/391 [01:23<00:00, 4.67it/s, loss=1.8234, acc=32.45%]
Train Loss: 1.8234 | Train Acc: 32.45%
Test Loss:  1.6543  | Test Acc:  38.92%
 New best test accuracy!

...

 Training Complete!
 Best Test Accuracy: 65.23%
⏱  Total Time: 415.32s (6.92 minutes)

 Verification:
 Custom Adam optimizer works with ResNet-18
 Deep network (18 layers) training successful
 Residual connections (skip connections) working
 Gradient flow through 11M parameters
```

---

##  Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size
- Find line: `BATCH_SIZE = 128`
- Change to: `BATCH_SIZE = 64` or `BATCH_SIZE = 32`

### Problem: "Module not found"
**Solution**: Everything should be self-contained
- Make sure you copied the **entire** script
- Check that all class definitions are present

### Problem: Slow training (CPU)
**Solution**: Enable GPU (see Step 3)
- Without GPU: ~2-3 hours
- With GPU: ~5-10 minutes

### Problem: Download fails
**Solution**: Enable Internet in settings
- Settings → Internet → ON

---

##  What This Proves

After successful run, this experiment demonstrates:

1.  **Deep Networks**: ResNet-18 has 18 layers (vs 3-5 in simple models)
2.  **Skip Connections**: Residual connections help gradient flow
3.  **Large Scale**: 11M parameters (vs 100K-1M in simple models)
4.  **Custom Optimizers**: Our Adam implementation works with modern architectures
5.  **Gradient Flow**: No vanishing gradients through 18 layers

This completes **Phase 12: Deep Model Architectures** 

---

##  Quick Copy Template

After running, fill this out:

```
KAGGLE RESULTS - ResNet-18 CIFAR-10
====================================

Device: [GPU/CPU name]
Training Time: [X minutes]
Best Test Accuracy: [X.XX%]

Epoch Results:
- Epoch 1: Train X.XX%, Test X.XX%
- Epoch 2: Train X.XX%, Test X.XX%
- Epoch 3: Train X.XX%, Test X.XX%
- Epoch 4: Train X.XX%, Test X.XX%
- Epoch 5: Train X.XX%, Test X.XX%

Status: [SUCCESS/FAILED]
Errors: [None or describe]

Verification:
 Custom optimizer worked: [YES/NO]
 Training completed: [YES/NO]
 No NaN/Inf values: [YES/NO]
```

---

##  After Completion

Once you share the results:
1. I'll verify the experiment succeeded
2. Update `docs/LIMITATIONS.md` Section 1.2
3. Mark Phase 12 as complete
4. Continue with next limitation

**Thank you for helping test on Kaggle!** 
