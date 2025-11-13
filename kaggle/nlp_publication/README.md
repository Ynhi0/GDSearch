# Kaggle NLP (IMDB) Publication Suite

This folder provides a Kaggle-ready, self-contained IMDB fine-tuning experiment suite using DistilBERT.
It runs multiple seeds across two optimizers (AdamW, SGD_Momentum), records wall-clock time and peak GPU memory,
and outputs publication-ready CSVs and a statistical comparison.

## What it does
- Downloads IMDB with `datasets`
- Tokenizes with `transformers` tokenizer
- Fine-tunes DistilBERT for sentiment classification
- Runs AdamW and SGD+Momentum across multiple seeds
- Logs per-epoch metrics and per-run telemetry
- Saves per-run CSVs and a stats CSV (Holm–Bonferroni corrected)

## Files
- `nlp_publication.py` — standalone script (no project imports)

## Outputs
Saved under `results/` (in Kaggle: `/kaggle/working/results`):
- `NN_DistilBERT_IMDB_<Optimizer>_lr<lr>_seed<seed>_publication.csv`
- `imdb_statistical_comparisons_publication.csv`

## Usage (Kaggle Notebook)
1) Enable GPU + Internet in Kaggle Notebook settings.
2) Optionally, install libraries if missing:
```python
!pip -q install transformers datasets accelerate
```
3) Run the suite:
```python
!python nlp_publication.py --quick
# or for full:
#!python nlp_publication.py --seeds 1,2,3,4,5 --epochs 3 --batch-size 16
```

### Resume support
You can pause and resume training. The script writes checkpoints each epoch under `--ckpt-dir` (default: `checkpoints`).

Resume an interrupted run:
```python
!python nlp_publication.py --seeds 1,2,3 --epochs 5 --batch-size 16 --resume --ckpt-dir checkpoints
```
