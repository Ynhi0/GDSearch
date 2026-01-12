#!/usr/bin/env python3
"""
Generate Statistical Report (MNIST, Analysis-ready)

Reads per-seed MNIST CSVs (produced by kaggle/mnist_benchmark/mnist_benchmark.py),
computes paired comparisons with normality checks, effect sizes, Holm-Bonferroni correction,
and basic power analysis. Outputs a CSV and a Markdown report.

Outputs:
  - results/nn_statistical_comparisons.csv
  - results/nn_statistical_report.md
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.statistical_analysis import power_analysis_report


OPTIMIZER_PATTERNS = {
    'SGD': 'MNIST_SimpleMLP_SGD_seed*.csv',
    'SGD_Momentum': 'MNIST_SimpleMLP_SGD_Momentum_seed*.csv',
    'Adam': 'MNIST_SimpleMLP_Adam_seed*.csv',
    'AdamW': 'MNIST_SimpleMLP_AdamW_seed*.csv',
    'AMSGrad': 'MNIST_SimpleMLP_AMSGrad_seed*.csv',
    'SAM_SGD': 'MNIST_SimpleMLP_SAM_SGD_seed*.csv',
    'SAM_Adam': 'MNIST_SimpleMLP_SAM_Adam_seed*.csv',
    'Lookahead_SGD': 'MNIST_SimpleMLP_Lookahead_SGD_seed*.csv',
    'Lookahead_Adam': 'MNIST_SimpleMLP_Lookahead_Adam_seed*.csv',
    'AdaBound': 'MNIST_SimpleMLP_AdaBound_seed*.csv',
    'RAdam': 'MNIST_SimpleMLP_RAdam_seed*.csv',
    'LAMB': 'MNIST_SimpleMLP_LAMB_seed*.csv',
}


from typing import Any

def _to_float(x: Any) -> float:
    """Safely coerce a value or array-like to a Python float.

    Handles tuples/lists, numpy scalars, 0-d arrays, 1-D arrays (taking first element),
    pandas Series/Index, and plain Python numbers/strings. Returns NaN on failure.
    """
    # Fast path for numeric scalars
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception as e:
        logging.debug("_to_float: fast-path type check failed for %r: %s", x, e, exc_info=True)

    # Pandas scalars/Series handling
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
    except Exception as e:
        logging.debug("_to_float: pandas branch failed for %r: %s", x, e, exc_info=True)

    # Unwrap common Python containers first
    try:
        if isinstance(x, (tuple, list)):
            if len(x) == 0:
                return float(np.nan)
            return _to_float(x[0])
    except Exception as e:
        logging.debug("_to_float: tuple/list branch failed for %r: %s", x, e, exc_info=True)

    # Numpy arrays and other array-likes
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float(np.nan)
        # If scalar or single-element, take that element
        if arr.shape == () or arr.size == 1:
            val = arr.item()
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
        # Otherwise prefer first element (recurse to handle nested containers)
        try:
            return _to_float(arr.ravel()[0])
        except Exception:
            return float(np.nan)
    except Exception as e:
        logging.debug("_to_float: numpy branch failed for %r: %s", x, e, exc_info=True)

    # String or other fallback - only call float on types that support it
    try:
        from src.utils.num_utils import safe_to_float
        if isinstance(x, (int, float, np.integer, np.floating, str)) or hasattr(x, "__float__"):
            return safe_to_float(x)
        # Try one-element array-like extraction as a last resort
        arr = np.asarray(x)
        if getattr(arr, "size", 0) == 1:
            try:
                return safe_to_float(arr.item())
            except Exception:
                return float(np.nan)
        return float(np.nan)
    except Exception:
        return float(np.nan)


def _load_final_metric(results_dir: str, optimizer: str, col: str) -> Dict[int, float]:
    pattern = str(Path(results_dir) / "experiments" / "mnist" / "experiments" / "mnist" / OPTIMIZER_PATTERNS[optimizer])
    data: Dict[int, float] = {}
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f)

            # Robust column selection: accept synonyms and prefer requested col when present
            col_candidates = [col]
            # Common alternate names
            if col.lower() in ("test_acc", "test_accuracy"):
                col_candidates += ["test_accuracy", "test_acc", "accuracy", "acc"]
            # Add any column that looks like a test accuracy/loss if not specified
            if not any(c in df.columns for c in col_candidates if c is not None):
                # look for column containing both 'test' and 'acc' or 'test' and 'loss'
                found = None
                for c in df.columns:
                    lc = c.lower()
                    if 'test' in lc and 'acc' in lc:
                        found = c
                        break
                    if 'test' in lc and 'loss' in lc and 'loss' in (col or ''):
                        found = c
                        break
                if found:
                    col_candidates.insert(0, found)

            chosen_col = None
            for c in col_candidates:
                if c in df.columns:
                    chosen_col = c
                    break

            if chosen_col is None:
                # No suitable column; skip file with a debug warning
                import logging
                logging.debug(f"Skipping file {f}: no column matching {col} or heuristics found; available columns: {list(df.columns)}")
                continue

            # Retrieve final value (prefer last epoch row) and coerce to float
            series_or_val = df[chosen_col]
            # If it's a Series-like, pick the last non-nan element safely
            val = np.nan
            try:
                if hasattr(series_or_val, 'dropna'):
                    s = series_or_val.dropna()
                    arr = np.asarray(s)
                    if arr.size == 0:
                        val = float(np.nan)
                    else:
                        last_arr = np.asarray(arr.ravel())
                        if last_arr.size == 0:
                            val = float(np.nan)
                        else:
                            last = last_arr.ravel()[-1]
                            # Unwrap common containers
                            if isinstance(last, (list, tuple, np.ndarray)):
                                try:
                                    last_val_arr = np.asarray(last).ravel()
                                    last_val = last_val_arr[-1] if last_val_arr.size > 0 else float(np.nan)
                                except Exception:
                                    last_val = last[0] if len(last) > 0 else float(np.nan)
                                val = _to_float(last_val)
                            else:
                                val = _to_float(last)
                else:
                    # ndarray or scalar
                    arr = np.asarray(series_or_val)
                    if arr.size == 0:
                        val = float(np.nan)
                    else:
                        val = _to_float(arr.ravel()[0])
            except Exception:
                # Fallback: try a best-effort extraction
                try:
                    if isinstance(series_or_val, (list, tuple)):
                        val = _to_float(series_or_val[0])
                    elif isinstance(series_or_val, np.ndarray) and series_or_val.size > 0:
                        val = _to_float(series_or_val.ravel()[0])
                    else:
                        val = _to_float(series_or_val)
                except Exception:
                    val = float(np.nan)

            # Extract seed id from filename (pattern: 'seed<digits>') as integer key
            seed = None
            try:
                import re
                m = re.search(r"seed(\d+)", Path(f).name)
                if m:
                    seed = int(m.group(1))
            except Exception:
                seed = None

            if seed is None:
                # fallback: use incremental index to avoid collisions
                seed = len(data) + 1

            data[seed] = float(val)
        except Exception as e:
            logging.debug("Error reading or parsing file %s: %s", f, e, exc_info=True)
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
        from src.analysis.statistical_analysis import safe_ttest_rel
        t_stat, p_val = safe_ttest_rel(a_vals, b_vals)
        diff = a_vals - b_vals
        eff = diff.mean() / (diff.std(ddof=1) + 1e-12)
        test = 'Paired t-test'
        effect_name = "Cohen's d"
    else:
        # Wilcoxon signed-rank
        W, p_val = stats.wilcoxon(a_vals, b_vals, zero_method='wilcox', correction=False, alternative='two-sided')
        # Some SciPy versions return scalars, others return scalar-like tuples; normalize to float
        if isinstance(W, (tuple, list, np.ndarray)):
            # Extract the statistic if a tuple-like is returned
            W_val = np.asarray(W).ravel()[0]
        else:
            W_val = W
        # Normalize W_val to a Python float
        W_val = _to_float(W_val)
        n = len(a_vals)
        # Rank-biserial correlation as effect size (use normalized W_val)
        eff = 1 - (2 * W_val) / (n * (n + 1))
        t_stat = np.nan
        test = 'Wilcoxon signed-rank'
        effect_name = 'Rank-biserial r'

    # Normalize and coerce numeric outputs to Python floats to satisfy callers and static type checks
    p_val_s = _to_float(p_val)

    def _sh_p(sh):
        try:
            if isinstance(sh, (tuple, list, np.ndarray)):
                return _to_float(sh[1])
            return _to_float(getattr(sh, 'pvalue', np.nan))
        except Exception:
            return float(np.nan)

    sh_a_p = _sh_p(sh_a)
    sh_b_p = _sh_p(sh_b)

    return {
        'name_A': str(name_a), 'name_B': str(name_b),
        'n': int(len(a_vals)),
        'mean_A': float(a_vals.mean()), 'std_A': float(a_vals.std(ddof=1)),
        'mean_B': float(b_vals.mean()), 'std_B': float(b_vals.std(ddof=1)),
        'test': str(test),
        'statistic': float(t_stat) if not np.isnan(t_stat) else np.nan,
        'p_value': p_val_s,
        'shapiro_p_A': sh_a_p,
        'shapiro_p_B': sh_b_p,
        'effect_size_name': str(effect_name),
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

    # Define comparison pairs - all pairwise comparisons
    optimizers = list(OPTIMIZER_PATTERNS.keys())
    pairs = [(a, b) for i, a in enumerate(optimizers) for b in optimizers[i+1:]]

    results = []
    for a, b in pairs:
        common = sorted(set(finals.get(a, {}).keys()) & set(finals.get(b, {}).keys()))
        if len(common) < 2:
            continue
        a_vals = np.array([finals[a][s] for s in common], dtype=float)
        b_vals = np.array([finals[b][s] for s in common], dtype=float)
        row = _paired_compare(a_vals, b_vals, a, b)

        # Power analysis (two-sided, alpha=0.05)
        try:
            # power_analysis_report expects results arrays, not pre-computed effect sizes
            power = power_analysis_report(
                results_A=a_vals,
                results_B=b_vals,
                name_A=a,
                name_B=b,
            )
            row['power_achieved'] = float(power['achieved_power'])
            row['n_for_80_power(d=obs)'] = int(power['n_required_for_80_power']) if power.get('n_required_for_80_power') else np.nan
        except Exception:
            row['power_achieved'] = np.nan
            row['n_for_80_power(d=obs)'] = np.nan

        results.append(row)

    if not results:
        logging.info("No valid comparisons found (need >=3 common seeds per pair).")
        return 1

    df = pd.DataFrame(results)
    # Holm-Bonferroni
    df['significant_holm'] = holm_bonferroni(df['p_value'].tolist())

    out_csv = Path(args.results_dir) / 'analysis' / 'nn_statistical_comparisons.csv'
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved: {out_csv}")
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
        sig_bool = bool(r.get('significant_holm', False))
        power_val = r.get('power_achieved', np.nan)
        lines.append(
            f"| {r['name_A']} | {r['name_B']} | {int(r['n'])} | {r['mean_A']:.4f} | {r['mean_B']:.4f} | {r['test']} | {r['p_value']:.3g} | {'✅' if sig_bool else '—'} | {r['effect_size_name']}={r['effect_size']:.3f} | {power_val:.2f} |"
        )
    report_path = Path(args.results_dir) / 'analysis' / 'nn_statistical_report.md'
    Path(report_path).write_text("\n".join(lines), encoding='utf-8')
    logging.info(f"Saved: {report_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
