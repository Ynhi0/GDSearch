# Kaggle MNIST Publication Experiments

This folder contains a fully self-contained MNIST experiment suite designed for Kaggle Notebooks. It runs multiple seeds across 7 optimizers and produces publication-ready CSV outputs and statistical comparisons.

## What it does
- Trains SimpleMLP on MNIST for 10 epochs per run
- Runs 7 optimizers × 10 seeds = 70 runs (configurable)
- Saves per-run CSVs and a statistical comparison CSV
- Uses paired tests when seeds match + Holm-Bonferroni correction

## Optimizers
- SGD (lr=0.01)
- SGD_Momentum (lr=0.05, momentum=0.9)
- Adam (lr=0.001)
- AdamW (lr=0.001, weight_decay=1e-4)
- AMSGrad (lr=0.001, amsgrad=True)
- SAM_SGD (lr=0.01, rho=0.05) - Sharpness-Aware Minimization with SGD base
- SAM_Adam (lr=0.001, rho=0.05) - Sharpness-Aware Minimization with Adam base

## Files
- `mnist_publication.py` — all-in-one script (self-contained; no project imports)
- `requirements.txt` — minimal dependencies (Kaggle usually preinstalls these)
- `QUICKSTART.md` — step-by-step instructions for Kaggle

## Output
Files are saved under `results/` (in Kaggle, this is `/kaggle/working/results`):
- `NN_SimpleMLP_MNIST_<Optimizer>_lr<lr>_seed<seed>_publication.csv`
- `mnist_statistical_comparisons_publication.csv`

## Tips
- Enable GPU in the notebook settings for fast training
- If Internet is disabled, enable it to automatically download MNIST via `torchvision`
- To shorten runtime, reduce `--epochs` or `--seeds`
