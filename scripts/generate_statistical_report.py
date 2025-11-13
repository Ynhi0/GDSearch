#!/usr/bin/env python3
"""
Generate Statistical Report (MNIST, Publication-ready)

Reads per-seed MNIST CSVs (produced by kaggle/mnist_publication/mnist_publication.py),
computes paired comparisons with normality checks, effect sizes, Holm-Bonferroni correction,
and basic power analysis. Outputs a CSV and a Markdown report.

Outputs:
  - results/nn_statistical_comparisons.csv
  - results/nn_statistical_report.md
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.statistical_analysis import power_analysis_report


OPTIMIZER_PATTERNS = {
    'SGD': 'NN_SimpleMLP_MNIST_SGD_*_publication.csv',
    'SGD_Momentum': 'NN_SimpleMLP_MNIST_SGD_Momentum_*_publication.csv',
    'Adam': 'NN_SimpleMLP_MNIST_Adam_*_publication.csv',
    'AdamW': 'NN_SimpleMLP_MNIST_AdamW_*_publication.csv',
    'AMSGrad': 'NN_SimpleMLP_MNIST_AMSGrad_*_publication.csv',
}


def _load_final_metric(results_dir: str, optimizer: str, col: str) -> Dict[int, float]:
    pattern = str(Path(results_dir) / OPTIMIZER_PATTERNS[optimizer])
    data: Dict[int, float] = {}
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f)
            val = float(df[col].iloc[-1])
            # Extract seed from filename
            import re
            m = re.search(r"seed(\d+)", f)
            if not m:
                continue
            seed = int(m.group(1))
            data[seed] = val
        except Exception:
            continue
    return data


def _paired_compare(a_vals: np.ndarray, b_vals: np.ndarray, name_a: str, name_b: str) -> Dict:
    # Normality diagnostics
    sh_a = stats.shapiro(a_vals) if len(a_vals) >= 3 else (np.nan, np.nan)
    sh_b = stats.shapiro(b_vals) if len(b_vals) >= 3 else (np.nan, np.nan)

    if (isinstance(sh_a, tuple) and isinstance(sh_b, tuple)
        and not np.isnan(sh_a[1]) and not np.isnan(sh_b[1])
        and sh_a[1] > 0.05 and sh_b[1] > 0.05):
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
        diff = a_vals - b_vals
        eff = diff.mean() / (diff.std(ddof=1) + 1e-12)
        test = 'Paired t-test'
        effect_name = "Cohen's d"
    else:
        # Wilcoxon signed-rank
        W, p_val = stats.wilcoxon(a_vals, b_vals, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
        n = len(a_vals)
        # Rank-biserial correlation as effect size
        eff = 1 - (2 * W) / (n * (n + 1))
        t_stat = np.nan
        test = 'Wilcoxon signed-rank'
        effect_name = 'Rank-biserial r'

    return {
        'name_A': name_a, 'name_B': name_b,
        'n': len(a_vals),
        'mean_A': float(a_vals.mean()), 'std_A': float(a_vals.std(ddof=1)),
        'mean_B': float(b_vals.mean()), 'std_B': float(b_vals.std(ddof=1)),
        'test': test,
        'statistic': float(t_stat) if not np.isnan(t_stat) else np.nan,
        'p_value': float(p_val),
        'shapiro_p_A': float(sh_a[1]) if isinstance(sh_a, tuple) else np.nan,
        'shapiro_p_B': float(sh_b[1]) if isinstance(sh_b, tuple) else np.nan,
        'effect_size_name': effect_name,
        'effect_size': float(eff),
    }


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
    parser = argparse.ArgumentParser(description='Generate Statistical Report (MNIST)')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--metric', type=str, default='test_acc', choices=['test_acc', 'test_loss'])
    args, _ = parser.parse_known_args()

    # Load per-optimizer, per-seed finals
    finals: Dict[str, Dict[int, float]] = {
        opt: _load_final_metric(args.results_dir, opt, args.metric)
        for opt in OPTIMIZER_PATTERNS.keys()
    }

    # Define comparison pairs
    pairs = [
        ('Adam', 'SGD'),
        ('SGD_Momentum', 'SGD'),
        ('RMSProp', 'SGD') if 'RMSProp' in finals else None,
        ('AdamW', 'Adam'),
        ('AMSGrad', 'Adam'),
        ('AdamW', 'SGD'),
        ('AMSGrad', 'AdamW'),
    ]
    pairs = [p for p in pairs if p is not None]

    results = []
    for a, b in pairs:
        common = sorted(set(finals.get(a, {}).keys()) & set(finals.get(b, {}).keys()))
        if len(common) < 3:
            continue
        a_vals = np.array([finals[a][s] for s in common], dtype=float)
        b_vals = np.array([finals[b][s] for s in common], dtype=float)
        row = _paired_compare(a_vals, b_vals, a, b)

        # Power analysis (two-sided, alpha=0.05)
        try:
            power = power_analysis_report(
                observed_effect_size=abs(row['effect_size']) if row['effect_size_name'] != "Cohen's d" else abs(row['effect_size']),
                n_samples=row['n'],
                name_A=a, name_B=b,
            )
            row['power_achieved'] = float(power['achieved_power'])
            row['n_for_80_power(d=obs)'] = int(power['n_required_for_80_power']) if power.get('n_required_for_80_power') else np.nan
        except Exception:
            row['power_achieved'] = np.nan
            row['n_for_80_power(d=obs)'] = np.nan

        results.append(row)

    if not results:
        print('No valid comparisons found (need >=3 common seeds per pair).')
        return 1

    df = pd.DataFrame(results)
    # Holm-Bonferroni
    df['significant_holm'] = holm_bonferroni(df['p_value'].tolist())

    out_csv = Path(args.results_dir) / 'nn_statistical_comparisons.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Markdown report
    lines = [
        "# MNIST Statistical Report",
        "",
        f"Metric: {args.metric}",
        "",
        "| Optimizer A | Optimizer B | n | Mean A | Mean B | Test | p-value | Holm sig | Effect | Power |",
        "|---|---:|---:|---:|---:|---|---:|:---:|---:|---:|",
    ]
    for _, r in df.sort_values('p_value').iterrows():
        lines.append(
            f"| {r['name_A']} | {r['name_B']} | {int(r['n'])} | {r['mean_A']:.4f} | {r['mean_B']:.4f} | {r['test']} | {r['p_value']:.3g} | {'✅' if r['significant_holm'] else '—'} | {r['effect_size_name']}={r['effect_size']:.3f} | {r.get('power_achieved', np.nan):.2f} |"
        )
    report_path = Path(args.results_dir) / 'nn_statistical_report.md'
    Path(report_path).write_text("\n".join(lines), encoding='utf-8')
    print(f"Saved: {report_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
