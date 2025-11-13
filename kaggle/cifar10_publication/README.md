# Kaggle CIFAR-10 Publication Experiments

This folder contains a fully self-contained CIFAR-10 experiment suite designed for Kaggle Notebooks. It runs multiple seeds across 5–6 optimizers and produces publication-ready CSV outputs and statistical comparisons.

## What it does
- Trains a small ConvNet on CIFAR-10 for N epochs per run
- Runs SGD, SGD+Momentum, RMSProp, Adam, AdamW, AMSGrad across multiple seeds
- Saves per-run CSVs and a statistical comparison CSV (Holm-Bonferroni corrected)

## Files
- `cifar10_publication.py` — all-in-one script (self-contained; no project imports)

## Output
Files are saved under `results/` (in Kaggle, this is `/kaggle/working/results`):
- `NN_SimpleCIFAR10_<Optimizer>_lr<lr>_seed<seed>_publication.csv`
- `cifar10_statistical_comparisons_publication.csv`

## Tips
- Enable GPU in the notebook settings for speed
- Enable Internet to download CIFAR-10 via `torchvision`
- Use `--quick` to shorten runtime