import os
import glob
import time
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from plot_results import plot_generalization_gap, plot_layer_grad_norms


RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'


def _list_csvs(prefix: str = '') -> List[str]:
    pattern = os.path.join(RESULTS_DIR, f"{prefix}*.csv")
    return sorted(glob.glob(pattern))


def _infer_kind(path: str) -> str:
    name = os.path.basename(path)
    if name.startswith('NN_'):
        return 'nn'
    return 'gd'


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_quantitative(threshold_loss: float = 1e-5, max_iters: int = 10**9) -> pd.DataFrame:
    rows = []
    for csv in _list_csvs():
        kind = _infer_kind(csv)
        df = load_df(csv)
        record = {
            'file': os.path.basename(csv),
            'kind': kind,
            'problem': None,
            'optimizer': None,
            'iters_to_thresh': None,
            'wall_time_to_thresh': None,  # not available (NA)
            'final_loss': None,
            'final_grad_norm': None,
            'test_accuracy_final': None,
            'generalization_gap_final': None,
        }
        if kind == 'gd':
            # Expect columns: iteration, loss, grad_norm
            record['problem'] = 'Rosenbrock/Misc'
            if 'optimizer' in df.columns:
                record['optimizer'] = df['optimizer'].iloc[0]
            # threshold
            if 'loss' in df.columns:
                hit = df.index[df['loss'] < threshold_loss]
                record['iters_to_thresh'] = int(hit[0]) if len(hit) > 0 else None
                record['final_loss'] = float(df['loss'].iloc[-1])
            if 'grad_norm' in df.columns:
                record['final_grad_norm'] = float(df['grad_norm'].iloc[-1])
        else:
            # NN: expect eval rows per epoch
            record['problem'] = 'MNIST/CIFAR-10'
            # infer optimizer from filename
            parts = os.path.basename(csv).split('_')
            if len(parts) > 3:
                record['optimizer'] = parts[3]
            eval_df = df[df.get('phase', '') == 'eval']
            train_df = df[df.get('phase', '') == 'train']
            meta_df = df[df.get('phase','') == 'meta']
            if not eval_df.empty:
                record['final_loss'] = float(eval_df['test_loss'].iloc[-1])
                record['test_accuracy_final'] = float(eval_df['test_accuracy'].iloc[-1])
                # gen gap final
                train_epoch_loss = train_df.groupby('epoch')['train_loss'].mean()
                ep = int(eval_df['epoch'].iloc[-1])
                tr_loss = float(train_epoch_loss.get(ep, np.nan))
                te_loss = float(eval_df['test_loss'].iloc[-1])
                record['generalization_gap_final'] = te_loss - tr_loss if not np.isnan(tr_loss) else None
                # Convergence metrics for NN
                if not meta_df.empty:
                    record['iters_to_thresh'] = int(meta_df['global_step'].iloc[0]) if 'global_step' in meta_df.columns else None
                    record['wall_time_to_thresh'] = float(meta_df['time_sec'].iloc[0]) if 'time_sec' in meta_df.columns else None
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_qualitative(nn_csv_paths: List[str]) -> pd.DataFrame:
    """
    Heuristic qualitative ratings based on metrics variability.
    - Smoothness: measured by std of curvature or grad_norm changes (proxy: rolling std of grad_norm). Lower std -> High smoothness.
    - Oscillation: measured by std of update_norm. Higher std -> High oscillation.
    - Hyperparameter Sensitivity: infer from filename grid (heuristic unknown -> Medium).
    - Saddle escape: use available 2D runs externally (not in NN) -> mark N/A for NN.
    """
    rows = []
    for path in nn_csv_paths:
        df = load_df(path)
        name = os.path.basename(path)
        train = df[df.get('phase', '') == 'train']
        # compute stds
        osc = float(train['update_norm'].std()) if 'update_norm' in train.columns and not train.empty else np.nan
        smth = float(train['grad_norm'].diff().abs().rolling(50).mean().mean()) if 'grad_norm' in train.columns and not train.empty else np.nan
        def bucket(value, thr_low, thr_high, invert=False):
            if np.isnan(value):
                return 'Unknown'
            if invert:
                # lower is better
                if value < thr_low:
                    return 'High'
                if value < thr_high:
                    return 'Medium'
                return 'Low'
            else:
                if value > thr_high:
                    return 'High'
                if value > thr_low:
                    return 'Medium'
                return 'Low'
        # Stricter thresholds calibrated for short MNIST runs (heuristic):
        smoothness = bucket(smth, 0.0005, 0.005, invert=True)  # lower diff -> smoother
        oscillation = bucket(osc, 0.02, 0.1, invert=False)      # higher std -> more oscillation
        sensitivity = 'Unknown'
        comment = ''
        if 'Adam' in name:
            comment = 'Converges fast; may plateau; stable early dynamics.'
        if 'SGD_Momentum' in name or 'SGD-Momentum' in name or 'SGD' in name:
            comment = 'Noisier updates; slower start; can generalize well.'
        rows.append({
            'algorithm': name,
            'trajectory_smoothness': smoothness,
            'oscillation_level': oscillation,
            'hyperparam_sensitivity': sensitivity,
            'saddle_escape': 'N/A',
            'notes': comment,
        })
    return pd.DataFrame(rows)


def generate_plots_for_nn(csv_path: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_df(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    plot_generalization_gap(df, title=f"Gen Gap & Test Acc: {base}", save_path=os.path.join(PLOTS_DIR, f"{base}_gen_gap.png"))
    plot_layer_grad_norms(df, title=f"Per-layer Grad Norms: {base}", save_path=os.path.join(PLOTS_DIR, f"{base}_layer_grads.png"))


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    qdf = summarize_quantitative()
    qdf.to_csv(os.path.join(RESULTS_DIR, 'summary_quantitative.csv'), index=False)

    nn_csvs = [p for p in _list_csvs('NN_')]
    if nn_csvs:
        qldf = summarize_qualitative(nn_csvs)
        qldf.to_csv(os.path.join(RESULTS_DIR, 'summary_qualitative.csv'), index=False)
        # also emit a markdown for readability
        qldf.to_markdown(os.path.join(RESULTS_DIR, 'summary_qualitative.md'), index=False)
        # Create plots for the first NN csv by default
        generate_plots_for_nn(nn_csvs[0])


if __name__ == '__main__':
    main()
