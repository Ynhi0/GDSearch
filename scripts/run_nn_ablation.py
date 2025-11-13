#!/usr/bin/env python3
"""
NN Ablation Study (Publication-ready)

Aggregates multi-seed MNIST results produced by kaggle/mnist_publication/mnist_publication.py
and generates publication-ready summaries and plots with error bars.

Outputs (default dirs results/ and plots/):
  - results/nn_ablation_summary.csv
  - plots/nn_ablation_accuracy.png
  - plots/nn_ablation_loss.png

Notes
- Designed to run after MNIST experiments finish (locally or on Kaggle).
- Uses final-epoch metrics from per-run CSVs.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OPTIMIZER_PATTERNS = {
    'SGD': 'NN_SimpleMLP_MNIST_SGD_*_publication.csv',
    'SGD_Momentum': 'NN_SimpleMLP_MNIST_SGD_Momentum_*_publication.csv',
    'Adam': 'NN_SimpleMLP_MNIST_Adam_*_publication.csv',
    'AdamW': 'NN_SimpleMLP_MNIST_AdamW_*_publication.csv',
    'AMSGrad': 'NN_SimpleMLP_MNIST_AMSGrad_*_publication.csv',
}


def _collect_runs(results_dir: str) -> Dict[str, List[pd.DataFrame]]:
    data: Dict[str, List[pd.DataFrame]] = {}
    for opt, pat in OPTIMIZER_PATTERNS.items():
        files = sorted(glob.glob(str(Path(results_dir) / pat)))
        runs: List[pd.DataFrame] = []
        for f in files:
            try:
                df = pd.read_csv(f)
                runs.append(df)
            except Exception:
                # Skip corrupted files
                continue
        data[opt] = runs
    return data


def _final_metric(dfs: List[pd.DataFrame], col: str) -> np.ndarray:
    vals: List[float] = []
    for df in dfs:
        if col in df.columns and len(df) > 0:
            vals.append(float(df[col].iloc[-1]))
    return np.array(vals, dtype=float)


def build_summary(results_dir: str = 'results') -> pd.DataFrame:
    data = _collect_runs(results_dir)
    rows = []
    for opt, runs in data.items():
        if not runs:
            continue
        acc = _final_metric(runs, 'test_acc')
        loss = _final_metric(runs, 'test_loss')
        rows.append({
            'Optimizer': opt,
            'n_runs': int(len(acc)),
            'test_acc_mean': float(np.nanmean(acc)) if acc.size else np.nan,
            'test_acc_std': float(np.nanstd(acc, ddof=1)) if acc.size > 1 else np.nan,
            'test_loss_mean': float(np.nanmean(loss)) if loss.size else np.nan,
            'test_loss_std': float(np.nanstd(loss, ddof=1)) if loss.size > 1 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values('Optimizer')
    return df


def _ensure_dirs(results_dir: str, plots_dir: str):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)


def plot_bars_with_error(df: pd.DataFrame, value_col: str, err_col: str, ylabel: str, title: str, save_path: str, invert_y: bool = False):
    if df.empty:
        print(f"Nothing to plot for {title}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    vals = df[value_col].values
    errs = df[err_col].values if err_col in df.columns else None
    bars = ax.bar(x, vals, yerr=errs, capsize=5, color=plt.cm.tab10(np.linspace(0, 1, len(df))), alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Optimizer'].tolist(), rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    if invert_y:
        ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(i, v + (0.002 if 'acc' in value_col else v * 0.02), f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NN Ablation Study (MNIST)')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--plots-dir', type=str, default='plots')
    args, _ = parser.parse_known_args()

    _ensure_dirs(args.results_dir, args.plots_dir)

    df = build_summary(args.results_dir)
    if df.empty:
        print("No runs found. Ensure MNIST publication experiments have been executed.")
        return 1
    out_csv = Path(args.results_dir) / 'nn_ablation_summary.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")

    # Accuracy plot (higher is better)
    plot_bars_with_error(
        df,
        value_col='test_acc_mean',
        err_col='test_acc_std',
        ylabel='Test Accuracy',
        title='MNIST Optimizer Ablation (mean ± std)',
        save_path=str(Path(args.plots_dir) / 'nn_ablation_accuracy.png'),
        invert_y=False,
    )

    # Loss plot (lower is better)
    plot_bars_with_error(
        df,
        value_col='test_loss_mean',
        err_col='test_loss_std',
        ylabel='Test Loss',
        title='MNIST Optimizer Ablation (mean ± std)',
        save_path=str(Path(args.plots_dir) / 'nn_ablation_loss.png'),
        invert_y=False,
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
