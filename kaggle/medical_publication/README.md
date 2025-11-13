# Kaggle Medical Segmentation Publication Suite (2D U-Net)

Lightweight, Kaggle-ready 2D segmentation experiments with a small U-Net in pure PyTorch.
Designed to run on user-provided 2D image/mask datasets under `/kaggle/input/<dataset>/images` and `/kaggle/input/<dataset>/masks`.

## What it does
- Loads 2D images and binary masks (PNG/JPG)
- Trains a small 2D U-Net for a few epochs across multiple seeds
- Compares Adam vs SGD+Momentum
- Logs Dice score and telemetry per run
- Saves per-run CSVs and a simple statistical comparison CSV

## Files
- `medical_publication.py` — standalone script (no repository imports)

## Inputs
Expected directory structure (example):
```
/kaggle/input/retina-seg/
  images/
    img001.png
    img002.png
    ...
  masks/
    img001.png
    img002.png
    ...
```
Masks should be single-channel binary (0/255 or 0/1); the script binarizes masks automatically.

## Outputs
Saved under `results/` (in Kaggle: `/kaggle/working/results`):
- `NN_UNet2D_MedSeg_<Optimizer>_seed<seed>_publication.csv`
- `medseg_statistical_comparisons_publication.csv`

## Usage (Kaggle Notebook)
1) Attach your dataset under `/kaggle/input/<dataset>`
2) Enable GPU
3) Run:
```python
!python medical_publication.py --data-root /kaggle/input/<dataset> --quick
# or full
#!python medical_publication.py --data-root /kaggle/input/<dataset> --seeds 1,2,3,4,5 --epochs 10
```
If no dataset is provided, the script falls back to a small synthetic dataset so you can validate the pipeline.

### Resume support
The script saves checkpoints each epoch (default dir: `checkpoints_medseg`). To resume after an interruption:
```python
!python medical_publication.py --data-root /kaggle/input/<dataset> --seeds 1,2,3 --epochs 10 --resume --ckpt-dir checkpoints_medseg
```
