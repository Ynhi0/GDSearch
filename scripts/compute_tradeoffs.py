#!/usr/bin/env python3
"""
Compute and visualize trade-offs:
- Accuracy vs Wall-Clock Time
- Accuracy vs Peak GPU Memory

Reads result CSVs in `results/` produced by Kaggle-ready scripts or internal runners.
Outputs summary CSVs and publication-ready plots in `plots/`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path('results')
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _collect_runs() -> pd.DataFrame:
    rows = []
    for p in RESULTS_DIR.glob('NN_*.csv'):
        try:
            df = pd.read_csv(p)
        except Exception:
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
                except Exception:
                    lr = None
            if part.startswith('seed'):
                try:
                    seed = int(part[4:])
                except Exception:
                    seed = None
        # Final metrics
        if 'test_acc' in df.columns:
            final_acc = float(df['test_acc'].iloc[-1])
        elif 'test_accuracy' in df.columns:
            final_acc = float(df['test_accuracy'].iloc[-1])
        else:
            final_acc = np.nan
        if 'elapsed_seconds' in df.columns:
            elapsed = float(df['elapsed_seconds'].iloc[-1])
        elif 'time_sec' in df.columns:
            elapsed = float(df['time_sec'].iloc[-1])
        else:
            elapsed = np.nan
        if 'peak_gpu_mb' in df.columns:
            peak_mb = float(df['peak_gpu_mb'].iloc[-1]) if pd.notna(df['peak_gpu_mb'].iloc[-1]) else np.nan
        elif 'peak_memory_MB' in df.columns:
            peak_mb = float(df['peak_memory_MB'].iloc[-1]) if pd.notna(df['peak_memory_MB'].iloc[-1]) else np.nan
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
    opts = sorted(df['optimizer'].dropna().unique())
    cmap = plt.cm.get_cmap('tab10', len(opts))
    for i, opt in enumerate(opts):
        sub = df[df['optimizer'] == opt]
        ax.scatter(sub[x], sub[y], label=opt, color=cmap(i), alpha=0.8)
    ax.set_xlabel(x.replace('_', ' ').title())
    ax.set_ylabel(y.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    df = _collect_runs()
    if df.empty:
        print("No result CSVs found in 'results/'.")
        return 1
    # Save summary
    summary_csv = RESULTS_DIR / 'tradeoffs_summary.csv'
    df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Per-dataset plots
    for dataset in sorted(df['dataset'].dropna().unique()):
        sub = df[df['dataset'] == dataset]
        if sub.empty:
            continue
        _scatter(sub, 'elapsed_seconds', 'final_test_acc', f"{dataset}: Accuracy vs Time", PLOTS_DIR / f"tradeoff_time_{dataset}.png")
        if sub['peak_gpu_mb'].notna().any():
            _scatter(sub, 'peak_gpu_mb', 'final_test_acc', f"{dataset}: Accuracy vs Peak GPU MB", PLOTS_DIR / f"tradeoff_memory_{dataset}.png")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
