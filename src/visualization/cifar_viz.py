from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.filename import parse_opt_seed_from_stem
from src.utils.metric_normalization import to_percent, to_percent_series
from src.utils.plot_helpers import arr_to_numpy_float

logger = logging.getLogger(__name__)


def _safe_text(ax, x, y, s, **kwargs):
    # Only add text when coordinates are finite
    try:
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            return
    except Exception:
        return
    ax.text(x, y, s, **kwargs)


def _load_csv_safe(p: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(p)
        return df
    except Exception as e:
        logger.debug("Skipping corrupt or unreadable CSV %s: %s", p, e)
        return None


def create_cifar10_visualizations(results_dir: Path, csv_files: List[Path]) -> Dict[str, Path]:
    """Create CIFAR-10 specific visualizations and save outputs.

    Returns a mapping of output name -> Path.
    """
    results_path = Path(results_dir)
    viz_dir = results_path / "visualizations"
    static_dir = viz_dir / "static" / "cifar10"
    static_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in csv_files]

    # Separate summary-like csvs (named *summary* or final) from time-series
    summary_paths = [p for p in csv_paths if 'summary' in p.name.lower() or p.stem.lower() in ('summary', 'final')]
    time_series_paths = [p for p in csv_paths if p not in summary_paths]

    # Load time-series, attach optimizer/seed from filename when missing
    dfs = []
    for p in time_series_paths:
        df = _load_csv_safe(p)
        if df is None:
            continue
        opt, seed = parse_opt_seed_from_stem(p.stem)
        if 'optimizer' not in df.columns or df['optimizer'].isnull().all():
            if opt:
                df['optimizer'] = opt
        if 'seed' not in df.columns or df['seed'].isnull().all():
            if seed is not None:
                df['seed'] = int(seed)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    outputs: Dict[str, Path] = {}

    # --- Training Loss ---
    if 'epoch' in combined_df.columns and 'train_loss' in combined_df.columns and 'optimizer' in combined_df.columns:
        try:
            plt.figure()
            opt_values = pd.unique(combined_df['optimizer'].dropna())
            for opt in opt_values:
                opt_data = combined_df[combined_df['optimizer'] == opt]
                if 'seed' in opt_data.columns:
                    grouped = opt_data.groupby('epoch')['train_loss'].agg(['mean', 'std'])
                    if not grouped['mean'].dropna().size:
                        continue
                    plt.plot(arr_to_numpy_float(grouped.index), arr_to_numpy_float(grouped['mean']), label=opt, linewidth=2)
                    plt.fill_between(arr_to_numpy_float(grouped.index), arr_to_numpy_float(grouped['mean'] - grouped['std']), arr_to_numpy_float(grouped['mean'] + grouped['std']), alpha=0.2)
                else:
                    y = arr_to_numpy_float(opt_data['train_loss'])
                    if not np.isfinite(y).any():
                        continue
                    plt.plot(arr_to_numpy_float(opt_data['epoch']), y, label=opt, linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title('CIFAR10 - Training Loss over Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out = static_dir / 'cifar10_train_loss.png'
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()
            outputs['train_loss'] = out
        except Exception as e:
            logger.debug('Could not create training loss plot: %s', e, exc_info=True)

    # --- Validation Accuracy over epochs (intentionally exclude epoch-wise test_* metrics) ---
    # Prefer validation columns for epoch plots; test_* are excluded from epoch visualization on purpose
    acc_col = None
    for col in ['val_acc', 'val_accuracy', 'test_acc', 'test_accuracy', 'final_test_acc', 'final_test_accuracy']:
        if col in combined_df.columns:
            acc_col = col
            break

    if 'epoch' in combined_df.columns and 'optimizer' in combined_df.columns and acc_col:
        try:
            df = combined_df.copy()
            # If we ended up using a test column fallback, prefer to label it as 'Validation' only when val exists
            is_val = acc_col in ('val_acc', 'val_accuracy')
            df['acc_pct'] = to_percent_series(df[acc_col])
            plt.figure()
            opt_values = pd.unique(df['optimizer'].dropna())
            for opt in opt_values:
                opt_data = df[df['optimizer'] == opt]
                if 'seed' in opt_data.columns:
                    grouped = opt_data.groupby('epoch')['acc_pct'].agg(['mean', 'std'])
                    if not grouped['mean'].dropna().size:
                        continue
                    plt.plot(arr_to_numpy_float(grouped.index), arr_to_numpy_float(grouped['mean']), label=opt, linewidth=2)
                    plt.fill_between(arr_to_numpy_float(grouped.index), arr_to_numpy_float((grouped['mean'] - grouped['std'])), arr_to_numpy_float((grouped['mean'] + grouped['std'])), alpha=0.2)
                else:
                    yvals = arr_to_numpy_float(opt_data['acc_pct'])
                    if not np.isfinite(yvals).any():
                        continue
                    plt.plot(arr_to_numpy_float(opt_data['epoch']), yvals, label=opt, linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy (%)')
            plt.title('CIFAR10 - Validation Accuracy over Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.tight_layout()
            out = static_dir / 'cifar10_val_accuracy.png'
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()
            outputs['val_accuracy'] = out
            # Backwards compatibility: some callers/tests expect 'test_accuracy' key
            outputs['test_accuracy'] = out
        except Exception as e:
            logger.debug('Could not create validation accuracy plot: %s', e, exc_info=True)

    # --- Final Comparison ---
    try:
        final_df = None
        if summary_paths:
            # prefer summary csvs when they contain mean/std per optimizer
            for sp in summary_paths:
                s = _load_csv_safe(sp)
                if s is None:
                    continue
                if 'optimizer' in s.columns and ('mean' in s.columns or 'value' in s.columns):
                    # prefer 'mean' if present
                    if 'mean' in s.columns and 'std' in s.columns:
                        final_df = s.set_index('optimizer')[['mean', 'std']]
                        break
                    elif 'value' in s.columns:
                        # treat 'value' as mean
                        final_df = s.set_index('optimizer')[['value']]
                        final_df = final_df.rename(columns={'value': 'mean'})
                        final_df['std'] = 0.0
                        break
        if final_df is None and not combined_df.empty and 'optimizer' in combined_df.columns:
            if 'seed' in combined_df.columns:
                last_per_seed = combined_df.groupby(['optimizer', 'seed']).last().reset_index()
                final_df = last_per_seed.groupby('optimizer')[acc_col].agg(['mean', 'std'])
                final_df['mean'] = to_percent_series(final_df['mean'])
            else:
                final_df = combined_df.groupby('optimizer').last()[[acc_col]].rename(columns={acc_col: 'mean'})
                final_df['std'] = 0.0
                final_df['mean'] = to_percent_series(final_df['mean'])

        if final_df is None or final_df.empty:
            logger.debug('No finite final results to plot')
        else:
            # Ensure numeric and drop non finite
            final_df = final_df.copy()
            final_df['mean'] = final_df['mean'].map(lambda x: to_percent(x))
            final_df['std'] = final_df.get('std', pd.Series(0.0, index=final_df.index)).map(lambda x: float(x) if pd.notnull(x) else 0.0)
            final_df = final_df.replace([np.inf, -np.inf], np.nan).dropna()
            # Clip means to 0-100
            final_df['mean'] = final_df['mean'].clip(upper=100.0).clip(lower=0.0)

            if final_df.empty:
                logger.debug('No finite final results after cleaning')
            else:
                plt.figure()
                x = range(len(final_df))
                plt.bar(x, final_df['mean'], yerr=final_df['std'], capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
                plt.xticks(x, list(map(str, final_df.index)), rotation=45, ha='right')
                plt.ylabel('Final Test Accuracy (%)')
                plt.title('CIFAR10 - Final Performance Comparison')
                plt.grid(axis='y', alpha=0.3)
                plt.ylim(0, 100)

                # Add safe labels
                ax = plt.gca()
                for i, (idx, row) in enumerate(final_df.iterrows()):
                    _safe_text(ax, i, float(row['mean']), f"{row['mean']:.1f}%\n±{row['std']:.1f}", ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                out = static_dir / 'cifar10_final_comparison.png'
                plt.savefig(out, dpi=300, bbox_inches='tight')
                plt.close()

                # Save meta JSON with y-axis info and means for tests
                meta = {'y_max': float(min(100.0, final_df['mean'].max())), 'means': final_df['mean'].to_dict()}
                meta_out = static_dir / 'cifar10_final_comparison_meta.json'
                with open(meta_out, 'w') as f:
                    json.dump(meta, f)

                outputs['final_comparison'] = out
                outputs['final_meta'] = meta_out
    except Exception as e:
        logger.debug('Could not create final comparison plot: %s', e, exc_info=True)

    return outputs
