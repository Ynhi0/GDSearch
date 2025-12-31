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


def _to_float(x: object) -> float:
    """Safely coerce a value or array-like to a Python float.

    Handles tuples/lists, numpy scalars, 0-d arrays, 1-D arrays (taking first element),
    pandas Series/Index, and plain Python numbers/strings. Returns NaN on failure.
    """
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        pass

    try:
        import pandas as _pd
        if isinstance(x, (_pd.Series, _pd.Index)):
            s = x.dropna()
            arr = np.asarray(s)
            if arr.size == 0:
                return float(np.nan)
            # take last non-NA element
            val = arr.ravel()[-1]
            return _to_float(val)
    except Exception:
        pass

    try:
        if isinstance(x, (tuple, list)):
            if len(x) == 0:
                return float(np.nan)
            return _to_float(x[0])
    except Exception:
        pass

    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float(np.nan)
        if arr.shape == () or arr.size == 1:
            val = arr.item()
            # Explicitly handle common scalar types to satisfy static checks
            if isinstance(val, (int, float, np.integer, np.floating, str)):
                try:
                    return float(val)
                except Exception:
                    return float(np.nan)
            if hasattr(val, "__float__"):
                try:
                    return float(val)
                except Exception:
                    return float(np.nan)
            return float(np.nan)
        # Non-scalar arrays: take first element and recurse
        try:
            return _to_float(arr.ravel()[0])
        except Exception:
            return float(np.nan)
    except Exception:
        pass

    try:
        from src.utils.num_utils import safe_to_float
        if isinstance(x, (int, float, np.integer, np.floating, str)) or hasattr(x, "__float__"):
            return safe_to_float(x)
        arr = np.asarray(x)
        if getattr(arr, "size", 0) == 1:
            try:
                return safe_to_float(arr.item())
            except Exception:
                return float(np.nan)
        return float(np.nan)
    except Exception:
        return float(np.nan)


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
            # Prefer requested metric column, but accept common synonyms
            chosen = None
            if metric in df.columns:
                chosen = metric
            elif metric == 'test_acc' and 'test_accuracy' in df.columns:
                chosen = 'test_accuracy'
            if chosen is None:
                continue

            last = df[chosen].dropna()
            if len(last) == 0:
                vals[seed] = float(np.nan)
                continue
            last_val = last.iloc[-1]
            vals[seed] = _to_float(last_val)
        except Exception as e:
            import logging
            logging.debug("Skipping unreadable or malformed CIFAR10 file %s: %s", f, e, exc_info=True)
            continue
    return vals


def _paired(valsA: np.ndarray, valsB: np.ndarray):
    # Normality
    pA = _to_float(stats.shapiro(valsA)[1]) if len(valsA) >= 3 else float(np.nan)
    pB = _to_float(stats.shapiro(valsB)[1]) if len(valsB) >= 3 else float(np.nan)

    if pA > 0.05 and pB > 0.05:
        stat, p = stats.ttest_rel(valsA, valsB)
        p = _to_float(p)
        effect_name = "Cohen's d"
        d = (valsA - valsB).mean() / (valsA - valsB).std(ddof=1)
        test = 'Paired t-test'
    else:
        W, p = stats.wilcoxon(valsA, valsB)
        # Normalize W (some SciPy versions return tuples)
        if isinstance(W, (tuple, list, np.ndarray)):
            W_val = np.asarray(W).ravel()[0]
        else:
            W_val = W
        W_val = _to_float(W_val)
        n = len(valsA)
        d = 1 - (2 * W_val) / (n * (n + 1))
        stat = float(np.nan)
        p = _to_float(p)
        test = 'Wilcoxon'
        effect_name = 'Rank-biserial r'

    # Ensure numeric return types
    return test, float(stat) if not np.isnan(stat) else np.nan, float(_to_float(p)), effect_name, float(d)


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
    from src.utils.num_utils import safe_to_float
    for _, r in df.sort_values('p_value').iterrows():
        sig_bool = bool(r.get('significant_holm', False))
        p_val = safe_to_float(r.get('p_value', np.nan))
        lines.append(
            f"| {r['name_A']} | {r['name_B']} | {int(r['n'])} | {r['mean_A']:.4f} | {r['mean_B']:.4f} | {r['test']} | {p_val:.3g} | {'✅' if sig_bool else '—'} | {r['effect_size_name']}={r['effect_size']:.3f} |"
        )
    report = Path(args.results_dir) / 'cifar10_statistical_report.md'
    Path(report).write_text('\n'.join(lines), encoding='utf-8')
    print(f"Saved: {report}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
