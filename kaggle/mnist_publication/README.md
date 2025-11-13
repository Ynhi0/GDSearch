# Kaggle MNIST Publication Experiments

This folder contains a fully self-contained MNIST experiment suite designed for Kaggle Notebooks. It runs multiple seeds across 5 optimizers and produces publication-ready CSV outputs and statistical comparisons.

## What it does
- Trains SimpleMLP on MNIST for 10 epochs per run
- Runs 5 optimizers × 10 seeds = 50 runs (configurable)
- Saves per-run CSVs and a statistical comparison CSV
- Uses paired tests when seeds match + Holm-Bonferroni correction

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
