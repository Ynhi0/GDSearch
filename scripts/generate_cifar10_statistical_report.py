#!/usr/bin/env python3
"""
Generate CIFAR-10 statistical report using paired tests and Holm–Bonferroni correction.
Reads per-seed CSVs produced by CIFAR-10 runners (internal or Kaggle-ready).
Outputs:
  - results/cifar10_statistical_comparisons.csv
  - results/cifar10_statistical_report.md
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


PATTERNS = {
    'SGD': 'NN_SimpleCIFAR10_SGD_*_seed*_*.csv',
    'SGD_Momentum': 'NN_SimpleCIFAR10_SGD_Momentum_*_seed*_*.csv',
    'RMSProp': 'NN_SimpleCIFAR10_RMSProp_*_seed*_*.csv',
    'Adam': 'NN_SimpleCIFAR10_Adam_*_seed*_*.csv',
    'AdamW': 'NN_SimpleCIFAR10_AdamW_*_seed*_*.csv',
    'AMSGrad': 'NN_SimpleCIFAR10_AMSGrad_*_seed*_*.csv',
}


def _load_final(results_dir: str, optimizer: str, metric: str) -> Dict[int, float]:
    pattern = str(Path(results_dir) / PATTERNS[optimizer])
    vals: Dict[int, float] = {}
    import re
    for f in glob.glob(pattern):
        m = re.search(r"seed(\d+)", f)
        if not m:
            continue
        seed = int(m.group(1))
        try:
            df = pd.read_csv(f)
            if metric in df.columns:
                vals[seed] = float(df[metric].iloc[-1])
            elif metric == 'test_acc' and 'test_accuracy' in df.columns:
                vals[seed] = float(df['test_accuracy'].iloc[-1])
        except Exception:
            continue
    return vals


def _paired(valsA: np.ndarray, valsB: np.ndarray):
    # Normality
    pA = stats.shapiro(valsA)[1] if len(valsA) >= 3 else np.nan
    pB = stats.shapiro(valsB)[1] if len(valsB) >= 3 else np.nan
    if pA > 0.05 and pB > 0.05:
        stat, p = stats.ttest_rel(valsA, valsB)
        effect_name = "Cohen's d"
        d = (valsA - valsB).mean() / (valsA - valsB).std(ddof=1)
        test = 'Paired t-test'
    else:
        W, p = stats.wilcoxon(valsA, valsB)
        n = len(valsA)
        d = 1 - (2 * W) / (n * (n + 1))
        stat = np.nan
        test = 'Wilcoxon'
        effect_name = 'Rank-biserial r'
    return test, float(stat) if not np.isnan(stat) else np.nan, float(p), effect_name, float(d)


def holm_bonferroni(pvals: List[float]) -> List[bool]:
    m = len(pvals)
    order = np.argsort(pvals)
    sig = [False] * m
    alpha = 0.05
    for k, idx in enumerate(order):
        if pvals[idx] < alpha / (m - k):
            sig[idx] = True
        else:
            break
    return sig


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CIFAR-10 Statistical Report')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--metric', type=str, default='test_acc')
    args, _ = parser.parse_known_args()

    finals = {opt: _load_final(args.results_dir, opt, args.metric) for opt in PATTERNS.keys()}

    pairs = [
        ('Adam', 'SGD'),
        ('AdamW', 'Adam'),
        ('AMSGrad', 'Adam'),
        ('SGD_Momentum', 'SGD'),
        ('RMSProp', 'SGD'),
        ('AdamW', 'SGD'),
        ('AMSGrad', 'SGD'),
        ('AMSGrad', 'AdamW'),
    ]

    rows = []
    for A, B in pairs:
        common = sorted(set(finals.get(A, {}).keys()) & set(finals.get(B, {}).keys()))
        if len(common) < 3:
            continue
        a = np.array([finals[A][s] for s in common])
        b = np.array([finals[B][s] for s in common])
        test, stat, p, eff_name, eff = _paired(a, b)
        rows.append({
            'name_A': A, 'name_B': B, 'n': len(common),
            'mean_A': float(a.mean()), 'std_A': float(a.std(ddof=1)),
            'mean_B': float(b.mean()), 'std_B': float(b.std(ddof=1)),
            'test': test, 'statistic': stat, 'p_value': p,
            'effect_size_name': eff_name, 'effect_size': eff,
        })

    if not rows:
        print('No valid comparisons (need >=3 common seeds).')
        return 1

    df = pd.DataFrame(rows)
    df['significant_holm'] = holm_bonferroni(df['p_value'].tolist())

    out_csv = Path(args.results_dir) / 'cifar10_statistical_comparisons.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    lines = [
        '# CIFAR-10 Statistical Report', '', f"Metric: {args.metric}", '',
        '| Optimizer A | Optimizer B | n | Mean A | Mean B | Test | p-value | Holm sig | Effect |',
        '|---|---:|---:|---:|---:|---|---:|:---:|---:|',
    ]
    for _, r in df.sort_values('p_value').iterrows():
        lines.append(
            f"| {r['name_A']} | {r['name_B']} | {int(r['n'])} | {r['mean_A']:.4f} | {r['mean_B']:.4f} | {r['test']} | {r['p_value']:.3g} | {'✅' if r['significant_holm'] else '—'} | {r['effect_size_name']}={r['effect_size']:.3f} |"
        )
    report = Path(args.results_dir) / 'cifar10_statistical_report.md'
    Path(report).write_text('\n'.join(lines), encoding='utf-8')
    print(f"Saved: {report}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
