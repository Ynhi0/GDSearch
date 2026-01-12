#!/usr/bin/env python3
"""
Compute and visualize trade-offs:
- Accuracy vs Wall-Clock Time
- Accuracy vs Peak GPU Memory

Reads result CSVs in `results/` produced by Kaggle-ready scripts or internal runners.
Outputs summary CSVs and benchmark-ready plots in `plots/`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

RESULTS_DIR = Path('results')
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _collect_runs() -> pd.DataFrame:
    rows = []
    for p in RESULTS_DIR.glob('NN_*.csv'):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logging.debug("Could not read result file %s: %s", p, e, exc_info=True)
            continue
        # Try to parse naming convention
        stem = p.stem
        parts = stem.split('_')
        if len(parts) < 4:
            continue
        model = parts[1]
        dataset = parts[2]
        optimizer = parts[3]
        lr = None
        seed = None
        for part in parts:
            if part.startswith('lr'):
                try:
                    lr = float(part[2:])
                except Exception as e:
                    logging.debug("Could not parse lr from %s: %s", part, e, exc_info=True)
                    lr = None
            if part.startswith('seed'):
                try:
                    seed = int(part[4:])
                except Exception as e:
                    logging.debug("Could not parse seed from %s: %s", part, e, exc_info=True)
                    seed = None
        # Final metrics
        from src.utils.num_utils import safe_to_float
        if 'test_acc' in df.columns:
            final_acc = safe_to_float(df['test_acc'])
        elif 'test_accuracy' in df.columns:
            final_acc = safe_to_float(df['test_accuracy'])
        else:
            final_acc = np.nan
        if 'elapsed_seconds' in df.columns:
            elapsed = safe_to_float(df['elapsed_seconds'])
        elif 'time_sec' in df.columns:
            elapsed = safe_to_float(df['time_sec'])
        else:
            elapsed = np.nan
        if 'peak_gpu_mb' in df.columns:
            peak_mb = safe_to_float(df['peak_gpu_mb'])
        elif 'peak_memory_MB' in df.columns:
            peak_mb = safe_to_float(df['peak_memory_MB'])
        else:
            peak_mb = np.nan

        rows.append({
            'file': p.name,
            'model': model,
            'dataset': dataset,
            'optimizer': optimizer,
            'lr': lr,
            'seed': seed,
            'final_test_acc': final_acc,
            'elapsed_seconds': elapsed,
            'peak_gpu_mb': peak_mb,
        })
    return pd.DataFrame(rows)


def _scatter(df: pd.DataFrame, x: str, y: str, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    # distinct markers/colors by optimizer
    opt_col = df.get('optimizer', np.array([]))
    if hasattr(opt_col, 'dropna'):
        opts = sorted(opt_col.dropna().unique())
    else:
        arr = np.asarray(opt_col)
        try:
            arr = arr[~pd.isna(arr)]
        except Exception as e:
            logging.debug("_scatter: filtering NA failed: %s", e, exc_info=True)
        opts = sorted(np.unique(arr).tolist())
    cmap = plt.cm.get_cmap('tab10', max(1, len(opts)))
    for i, opt in enumerate(opts):
        sub = df[df['optimizer'] == opt]
        sub = pd.DataFrame(sub)
        ax.scatter(sub[x], sub[y], label=opt, color=cmap(i), alpha=0.8)
    ax.set_xlabel(x.replace('_', ' ').title())
    ax.set_ylabel(y.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    logging.info("Saved: %s", out_png)


def main():
    df = _collect_runs()
    if df.empty:
        logging.info("No result CSVs found in 'results'.")
        return 1
    # Save summary
    summary_csv = RESULTS_DIR / 'tradeoffs_summary.csv'
    df.to_csv(summary_csv, index=False)
    logging.info("Saved: %s", summary_csv)

    # Per-dataset plots
    ds_col = df.get('dataset', np.array([]))
    if hasattr(ds_col, 'dropna'):
        ds_list = sorted(ds_col.dropna().unique())
    else:
        arr = np.asarray(ds_col)
        try:
            arr = arr[~pd.isna(arr)]
        except Exception as e:
            logging.debug("compute_tradeoffs: filtering NA failed for ds_col: %s", e, exc_info=True)
        ds_list = sorted(np.unique(arr).tolist())

    for dataset in ds_list:
        sub = df[df['dataset'] == dataset]
        if sub.empty:
            continue
        sub = pd.DataFrame(sub)
        _scatter(sub, 'elapsed_seconds', 'final_test_acc', f"{dataset}: Accuracy vs Time", PLOTS_DIR / f"tradeoff_time_{dataset}.png")
        col = sub.get('peak_gpu_mb', np.array([]))
        arr = np.asarray(col)
        if arr.size > 0 and np.any(~pd.isna(arr)):
            _scatter(pd.DataFrame(sub), 'peak_gpu_mb', 'final_test_acc', f"{dataset}: Accuracy vs Peak GPU MB", PLOTS_DIR / f"tradeoff_memory_{dataset}.png")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
