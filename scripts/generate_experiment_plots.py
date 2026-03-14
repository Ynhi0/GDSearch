#!/usr/bin/env python3
"""
Universal visualization generator for all experiments.
Reads CSV files from results/ and automatically generates high-quality plots.
"""

import os
import sys
import glob
import shutil
import hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable, Any
from datetime import datetime, timezone
import seaborn as sns
import traceback
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

STATIC_ONLY_EXPERIMENTS = {
    "2d_optimization",
    "2d_visualization",
    "robustness",
    "sam_sensitivity",
    "ablation",
    "beta_sensitivity_2d",
    "hyperparam_sensitivity",
    "hyperparameter_heatmaps",
    "convergence_validation",
    "saddle_escape",
    "stochastic_2d_integrity",
    "adam_adamw_comparison",
    "optimizer_comparison",
    "cross_optimizer_dynamics",
    "beta_sensitivity_training",
    "theory_practice",
}

SKIP_DIR_PARTS = {
    "analysis",
    "reports",
    "visualizations",
    "final_deliverables",
    "checkpoints",
}

META_FILENAME_PATTERNS = [
    "summary",
    "aggregation",
    "manifest",
    "heatmap_data",
    "convergence_rate",
    "cross_experiment",
    "statistical",
    "comparison_matrix",
    "results",  # exact/near-exact aggregate result files
    # QC FIX: trajectory/experiment-specific CSVs have no loss/acc column headers
    # but are valid outputs — treat them as metadata, not flagged no_metric_cols.
    "trajectory",       # optimizer trajectory CSVs (x, y, loss, grad_norm)
    "stochastic_2d",    # stochastic 2D integrity experiment outputs
    "saddle_escape",    # saddle point escape experiment summaries
    "gd_vs_sgd",        # GD vs SGD comparison outputs
]

META_EXPERIMENTS = {
    "2d_visualization",
    "hyperparameter_heatmaps",
    "optimizer_comparison",
    "theory_practice",
}


def _is_metadata_csv(csv_path: Path, experiment: str) -> bool:
    name_lc = csv_path.name.lower()
    if any(pattern in name_lc for pattern in META_FILENAME_PATTERNS):
        return True
    if experiment in META_EXPERIMENTS:
        return True
    return False

# Families prone to stale duplicates across result roots.
PNG_DUPLICATE_PREFIXES = (
    "2d_",
    "robustness_",
    "sam_",
    "ablation_",
    "momentum_",
    "adam_",
    "optimizer_comparison_",
)
STALE_GENERIC_PREFIXES = (
    "2d_",
    "2d_visualization_",
    "robustness_",
    "sam_sensitivity_",
    "ablation_",
    "beta_sensitivity_2d_",
    "hyperparam_sensitivity_",
    "hyperparameter_heatmaps_",
    "convergence_validation_",
    "saddle_escape_",
    "stochastic_2d_integrity_",
    "adam_adamw_comparison_",
    "optimizer_comparison_",
    "cross_optimizer_dynamics_",
    "beta_sensitivity_training_",
    "theory_practice_",
)
ALLOWED_GENERIC_PREFIXES = (
    "mnist_",
    "cifar10_",
    "imdb_",
    "medical_",
    "advanced_ablation_",
    "init_ablation_",
    "lr_ablation_",
    "wd_ablation_",
    "scheduler_ablation_",
    "missing_ablations_",
    "convergence_validation_",
    "saddle_escape_",
    "adam_adamw_comparison_",
    "hyperparam_sensitivity_",
    "hyperparameter_heatmaps_",
    "batch_ablation_",
    "optimizer_comparison_",
    "cross_optimizer_dynamics_",
    "beta_sensitivity_training_",
    "theory_practice_",
)

PROPOSAL_REQUIRED_PATTERNS: Dict[str, List[str]] = {
    "2d_optimization": [
        "visualizations/static/2d_optimization/2d_rastrigin_loss_convergence.png",
        "visualizations/static/2d_optimization/2d_rastrigin_grad_norm_convergence.png",
        "visualizations/static/2d_optimization/2d_rastrigin_trajectory_overlay.png",
        "visualizations/static/2d_optimization/2d_rosenbrock_loss_convergence.png",
        "visualizations/static/2d_optimization/2d_rosenbrock_grad_norm_convergence.png",
        "visualizations/static/2d_optimization/2d_rosenbrock_trajectory_overlay.png",
        "visualizations/static/2d_optimization/2d_final_loss_by_optimizer.png",
        "visualizations/static/2d_optimization/2d_convergence_rate_by_optimizer.png",
        "visualizations/static/2d_optimization/2d_convergence_rate_strict_by_optimizer.png",
        "visualizations/static/2d_optimization/2d_convergence_profile_by_threshold.png",
    ],
    "robustness": [
        "visualizations/static/robustness/robustness_loss_convergence.png",
        "visualizations/static/robustness/robustness_grad_norm_convergence.png",
        "visualizations/static/robustness/robustness_start_point_sensitivity.png",
        "visualizations/static/robustness/robustness_final_loss_by_optimizer.png",
        "visualizations/static/robustness/robustness_convergence_rate.png",
        "visualizations/static/robustness/robustness_convergence_rate_strict.png",
        "visualizations/static/robustness/robustness_convergence_profile.png",
    ],
    "sam_sensitivity": [
        "visualizations/static/sam_sensitivity/sam_rho_sweep.png",
        "visualizations/static/sam_sensitivity/sam_all_metrics.png",
        "visualizations/static/sam_sensitivity/sam_train_loss_by_epoch.png",
        "visualizations/static/sam_sensitivity/sam_test_accuracy_by_epoch.png",
    ],
    "ablation": [
        "visualizations/static/ablation/ablation_loss_convergence.png",
        "visualizations/static/ablation/ablation_final_loss_by_optimizer.png",
        "visualizations/static/ablation/ablation_iterations_by_optimizer.png",
        "visualizations/static/ablation/ablation_convergence_rate.png",
        "visualizations/static/ablation/ablation_convergence_profile.png",
    ],
    "beta_sensitivity_2d": [
        "experiments/beta_sensitivity_2d/rosenbrock/momentum/momentum_trajectories.png",
        "experiments/beta_sensitivity_2d/rosenbrock/momentum/momentum_metrics.png",
        "experiments/beta_sensitivity_2d/saddle_point/adam/adam_heatmaps.png",
    ],
}

PROPOSAL_REQUIRED_METRICS: Dict[str, Dict[str, Any]] = {
    "2d_optimization": {
        "summary_csv": "experiments/2d_optimization/2d_optimization_results.csv",
        "expected_unique_seeds": 10,
        "required_columns": [
            "optimizer",
            "function",
            "seed",
            "final_loss",
            "iterations",
            "run_status",
            "error",
            "converged_strict",
            "converged_practical",
        ],
        "at_least_one_metric_cols": ["final_grad_norm", "gradient_norm"],
        "at_least_one_convergence_cols": [
            "converged_epoch",
            "converged_iteration",
            "converged_iteration_strict",
            "converged_iteration_practical",
        ],
    },
    "robustness": {
        "summary_csv": "experiments/robustness/robustness_results.csv",
        "expected_unique_seeds": 10,
        "required_columns": [
            "optimizer",
            "seed",
            "start_point",
            "final_loss",
            "iterations",
            "run_status",
            "error",
            "converged",
            "converged_iteration",
        ],
        "at_least_one_metric_cols": ["final_grad_norm", "gradient_norm"],
        "at_least_one_convergence_cols": ["converged_epoch", "converged_iteration"],
    },
    "sam_sensitivity": {
        "summary_csv": "experiments/sam_sensitivity/sam_sensitivity_results.csv",
        "expected_unique_seeds": 10,
        "required_columns": [
            "rho",
            "seed",
            "final_loss",
            "final_train_loss",
            "final_test_accuracy",
            "epochs_trained",
        ],
        "at_least_one_metric_cols": ["final_test_loss", "test_acc", "best_test_accuracy"],
        "at_least_one_convergence_cols": [],
    },
    "ablation": {
        "summary_csv": "experiments/ablation/ablation_results.csv",
        "expected_unique_seeds": 10,
        "required_columns": [
            "optimizer",
            "seed",
            "final_loss",
            "iterations",
            "run_status",
            "error",
            "converged_strict",
            "converged_practical",
        ],
        "at_least_one_metric_cols": ["final_grad_norm", "gradient_norm"],
        "at_least_one_convergence_cols": [
            "converged_epoch",
            "converged_iteration",
            "converged_iteration_strict",
            "converged_iteration_practical",
        ],
    },
}

PROPOSAL_EXPECTATION_NOTES = [
    "Convergence criterion for optimizer-level summary plots uses stationarity-or-loss: (||grad|| < t) OR (loss < t).",
    "Practical threshold is t = 1e-3 and strict threshold is t = 1e-6 for 2D/robustness/ablation families.",
    "Convergence-profile denominator uses all attempted runs (error/non-finite rows count as non-converged).",
    "Error rows (run_status starts with 'error' or non-empty error column) are always forced non-converged.",
    "Convergence-rate analysis is based on optimization signals (loss/grad), not test accuracy curves.",
]

PROPOSAL_SOURCE_DOCS = [
    "docs/CONVERGENCE_CRITERIA.md",
    "docs/METRICS_HIERARCHY.md",
    "docs/proposal.txt",
]

ALL_NULL_ALLOWLIST = {
    "error",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_png_files_anycase(root: Path) -> Iterable[Path]:
    """Yield PNG files regardless of extension case (.png/.PNG)."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".png":
            yield p


def _is_meta_csv(path_obj: Path) -> bool:
    stem = path_obj.stem.lower()
    if stem in {"results", "summary", "manifest"}:
        return True
    return any(pat in stem for pat in META_FILENAME_PATTERNS)


def _is_experiment_csv(path_obj: Path) -> bool:
    parts = [p.lower() for p in path_obj.parts]
    if "experiments" not in parts:
        return False
    if any(part in SKIP_DIR_PARTS for part in parts):
        return False
    return True


def _extract_experiment_name(path_obj: Path) -> Optional[str]:
    parts = list(path_obj.parts)
    lower_parts = [p.lower() for p in parts]
    if "experiments" not in lower_parts:
        return None
    idx = lower_parts.index("experiments")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def _finite_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce")
    return int(np.isfinite(values.to_numpy(dtype=float)).sum())


def _as_bool_series(series: pd.Series) -> pd.Series:
    txt = series.astype(str).str.lower().str.strip()
    num = pd.to_numeric(series, errors='coerce')
    return txt.isin({'1', 'true', 't', 'yes', 'y'}) | (num.fillna(0) != 0)


def backfill_convergence_columns(results_dir: Path) -> Path:
    """
    Backfill converged_epoch/run_status aliases in legacy result CSVs.

    This prevents empty convergence-epoch aggregations when older CSVs only have
    converged_iteration (or strict/practical variants).
    """
    rows = []
    for csv_path in results_dir.rglob("*.csv"):
        if not csv_path.is_file() or not _is_experiment_csv(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            rows.append({"path": str(csv_path), "action": f"read_error: {e}"})
            continue
        if df.empty:
            rows.append({"path": str(csv_path), "action": "skip_empty"})
            continue

        cols_lc = {str(c).lower().strip() for c in df.columns}
        has_convergence_markers = any(
            c in cols_lc
            for c in (
                'converged',
                'converged_strict',
                'converged_practical',
                'converged_iteration',
                'converged_iteration_strict',
                'converged_iteration_practical',
                'converged_step',
                'converged_epoch',
            )
        )
        has_summary_shape = {'final_loss', 'iterations'}.issubset(cols_lc)
        if not (has_convergence_markers or has_summary_shape):
            rows.append({"path": str(csv_path), "action": "skip_non_summary"})
            continue

        changed = False

        if 'run_status' in df.columns:
            status = df['run_status'].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'none': np.nan})
            status = status.fillna('max_iters_or_stalled')
        else:
            status = pd.Series('max_iters_or_stalled', index=df.index, dtype=object)
        status_before = status.copy()
        if True:
            err = df.get('error', pd.Series('', index=df.index))
            err_txt = err.astype(str).str.lower().str.strip()
            has_error = ~err_txt.isin({'', 'nan', 'none'})
            non_finite_err = err_txt.str.contains('non-finite|nonfinite|inf|nan', regex=True)
            status.loc[has_error] = np.where(non_finite_err.loc[has_error], 'error_non_finite_loss', 'error_exception')

            if 'converged_strict' in df.columns:
                strict = _as_bool_series(df['converged_strict'])
                status.loc[strict & ~has_error] = 'converged_strict'
            if 'converged_practical' in df.columns:
                practical = _as_bool_series(df['converged_practical'])
                mask = practical & ~has_error & (~status.eq('converged_strict'))
                status.loc[mask] = 'converged_practical'
            if 'converged' in df.columns:
                conv = _as_bool_series(df['converged'])
                status.loc[conv & ~has_error & status.eq('max_iters_or_stalled')] = 'converged'
            if 'final_loss' in df.columns:
                final_loss = pd.to_numeric(df['final_loss'], errors='coerce')
                status.loc[~np.isfinite(final_loss) & ~has_error] = 'error_non_finite_loss'

            df['run_status'] = status.astype(str)
            if not status.equals(status_before):
                changed = True

        # Build/repair converged_epoch from available convergence hints.
        epoch_before = pd.to_numeric(df['converged_epoch'], errors='coerce') if 'converged_epoch' in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
        epoch = epoch_before.copy()
        candidate_cols = [
            'converged_step',
            'converged_iteration_strict',
            'converged_iteration_practical',
            'converged_iteration',
            'iterations_to_converge',
        ]
        for c in candidate_cols:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors='coerce')
                mask = epoch.isna() & np.isfinite(vals) & (vals >= 0)
                if mask.any():
                    epoch.loc[mask] = vals.loc[mask]

        if {'converged', 'iterations'}.issubset(df.columns):
            conv = _as_bool_series(df['converged'])
            iters = pd.to_numeric(df['iterations'], errors='coerce')
            derived = (iters - 1).clip(lower=0)
            mask = epoch.isna() & conv & np.isfinite(derived)
            if mask.any():
                epoch.loc[mask] = derived.loc[mask]

        # Replace empty convergence epochs on clearly non-converged runs with -1.
        non_conv = df['run_status'].astype(str).str.startswith('max_iters') | (df['run_status'].astype(str) == 'converged_false')
        if 'converged' in df.columns:
            conv = _as_bool_series(df['converged'])
            non_conv = non_conv | (~conv)
        fill_mask = epoch.isna() & non_conv
        if fill_mask.any():
            epoch.loc[fill_mask] = -1.0

        if ('converged_epoch' not in df.columns) or (not epoch.equals(epoch_before)):
            df['converged_epoch'] = epoch
            changed = True

        if 'converged_epoch' in df.columns:
            conv_iter_before = pd.to_numeric(df['converged_iteration'], errors='coerce') if 'converged_iteration' in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
            conv_iter = conv_iter_before.copy()
            epoch_num = pd.to_numeric(df['converged_epoch'], errors='coerce')
            fill_iter = conv_iter.isna() & np.isfinite(epoch_num)
            if fill_iter.any():
                conv_iter.loc[fill_iter] = epoch_num.loc[fill_iter]
            if ('converged_iteration' not in df.columns) or (not conv_iter.equals(conv_iter_before)):
                df['converged_iteration'] = conv_iter
                changed = True

        # Normalize convergence marker columns to avoid large NaN-only fields in summary CSVs.
        for marker_col in ('converged_epoch', 'converged_iteration', 'converged_iteration_strict', 'converged_iteration_practical'):
            if marker_col in df.columns:
                marker_before = pd.to_numeric(df[marker_col], errors='coerce')
                marker_after = marker_before.fillna(-1.0)
                if not marker_after.equals(marker_before):
                    df[marker_col] = marker_after
                    changed = True

        if changed:
            df.to_csv(csv_path, index=False)
            rows.append({"path": str(csv_path), "action": "updated"})
        else:
            rows.append({"path": str(csv_path), "action": "unchanged"})

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / "convergence_column_backfill_report.csv"
    pd.DataFrame(rows, columns=["path", "action"]).to_csv(report_path, index=False)
    updated = sum(str(r.get("action", "")).startswith("updated") for r in rows)
    print(f"[BACKFILL] Updated {updated} CSV files with convergence aliases/status")
    print(f"[BACKFILL] Report: {report_path}")
    return report_path


def _looks_like_training_timeseries(df: pd.DataFrame, min_len: int = 5) -> bool:
    if df is None or df.empty or len(df) < min_len:
        return False

    cols = [str(c).lower().strip() for c in df.columns]
    has_axis = any(("epoch" in c) or ("iter" in c) for c in cols)
    if not has_axis:
        return False

    metric_cols = []
    for c in df.columns:
        cl = str(c).lower()
        if any(token in cl for token in ["loss", "acc", "accuracy", "dice"]):
            if not any(meta_token in cl for meta_token in ["final_", "mean_", "std_", "summary", "rate", "count"]):
                metric_cols.append(c)
    if not metric_cols:
        return False

    for mc in metric_cols:
        finite_vals = pd.to_numeric(df[mc], errors="coerce").dropna()
        if len(finite_vals) >= min_len and np.isfinite(finite_vals.to_numpy(dtype=float)).any():
            return True
    return False


def _write_status_plot(status_csv: Path, output_path: Path, title: str) -> bool:
    if not status_csv.exists():
        return False
    try:
        df = pd.read_csv(status_csv)
    except Exception:
        return False
    if df.empty:
        return False

    row = df.iloc[0].to_dict()
    lines = [title]
    status = str(row.get("status", "unknown")).strip()
    reason = str(row.get("reason", "")).strip()
    required = str(row.get("required_input_dirs", "")).strip()
    if status:
        lines.append(f"Status: {status}")
    if reason:
        lines.append(f"Reason: {reason}")
    if required:
        lines.append(f"Required inputs: {required}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=11,
        family="monospace",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved status plot: {output_path}")
    return True


def _maybe_generate_theory_practice_status_plot(results_dir: Path) -> None:
    status_csv = results_dir / "experiments" / "theory_practice" / "theory_practice_status.csv"
    if not status_csv.exists():
        return
    output_path = results_dir / "visualizations" / "theory_practice_status.png"
    _write_status_plot(status_csv, output_path, "Theory-Practice Validation")


def _canonical_priority(path_obj: Path) -> int:
    path_str = str(path_obj).replace("\\", "/").lower()
    if "/visualizations/static/" in path_str:
        return 100
    if "/experiments/2d_visualization/" in path_str:
        return 90
    if "/experiments/beta_sensitivity_2d/" in path_str:
        return 80
    return 10


def overwrite_duplicate_pngs(results_dir: Path) -> Path:
    """One-time duplicate overwrite pass for proposal-critical PNG families."""
    png_files = list(_iter_png_files_anycase(results_dir))
    groups: Dict[str, List[Path]] = {}
    for p in png_files:
        name = p.name.lower()
        if not name.startswith(PNG_DUPLICATE_PREFIXES):
            continue
        groups.setdefault(name, []).append(p)

    sync_rows = []
    overwritten = 0
    for filename_lc, paths in groups.items():
        if len(paths) <= 1:
            continue
        filename = paths[0].name
        canonical = sorted(
            paths,
            key=lambda p: (_canonical_priority(p), p.stat().st_mtime),
            reverse=True,
        )[0]
        canon_hash = _sha256_file(canonical)

        for target in paths:
            if target == canonical:
                continue
            try:
                target_hash = _sha256_file(target)
                if target_hash != canon_hash:
                    shutil.copy2(canonical, target)
                    overwritten += 1
                sync_rows.append(
                    {
                        "filename": filename,
                        "filename_lc": filename_lc,
                        "canonical": str(canonical),
                        "target": str(target),
                        "action": "overwritten" if target_hash != canon_hash else "already_synced",
                    }
                )
            except Exception as e:
                sync_rows.append(
                    {
                        "filename": filename,
                        "filename_lc": filename_lc,
                        "canonical": str(canonical),
                        "target": str(target),
                        "action": f"error: {e}",
                    }
                )

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / "canonical_png_sync_report.csv"
    report_cols = ["filename", "filename_lc", "canonical", "target", "action"]
    pd.DataFrame(sync_rows, columns=report_cols).to_csv(report_path, index=False)
    print(f"[SYNC] Overwrote {overwritten} duplicate PNGs")
    print(f"[SYNC] Report: {report_path}")
    return report_path


def overwrite_duplicate_pngs_across_roots(canonical_results_dir: Path, peer_dirs: List[Path]) -> Path:
    """
    Overwrite duplicate proposal-critical PNGs in peer trees from one canonical tree.
    """
    canonical_files = [
        p for p in _iter_png_files_anycase(canonical_results_dir)
        if p.name.lower().startswith(PNG_DUPLICATE_PREFIXES)
    ]
    canonical_by_name: Dict[str, Path] = {}
    for p in canonical_files:
        name_lc = p.name.lower()
        current = canonical_by_name.get(name_lc)
        if current is None:
            canonical_by_name[name_lc] = p
            continue
        prev_score = (_canonical_priority(current), current.stat().st_mtime)
        new_score = (_canonical_priority(p), p.stat().st_mtime)
        if new_score > prev_score:
            canonical_by_name[name_lc] = p

    rows = []
    overwritten = 0
    for peer_root in peer_dirs:
        if not peer_root.exists():
            rows.append({
                "peer_root": str(peer_root),
                "filename": "",
                "canonical": "",
                "target": "",
                "action": "missing_peer_root",
            })
            continue
        for target in _iter_png_files_anycase(peer_root):
            if not target.name.lower().startswith(PNG_DUPLICATE_PREFIXES):
                continue
            canonical = canonical_by_name.get(target.name.lower())
            if canonical is None:
                rows.append({
                    "peer_root": str(peer_root),
                    "filename": target.name,
                    "canonical": "",
                    "target": str(target),
                    "action": "no_canonical_match",
                })
                continue
            try:
                canon_hash = _sha256_file(canonical)
                target_hash = _sha256_file(target)
                if canon_hash != target_hash:
                    shutil.copy2(canonical, target)
                    overwritten += 1
                    action = "overwritten"
                else:
                    action = "already_synced"
                rows.append({
                    "peer_root": str(peer_root),
                    "filename": target.name,
                    "canonical": str(canonical),
                    "target": str(target),
                    "action": action,
                })
            except Exception as e:
                rows.append({
                    "peer_root": str(peer_root),
                    "filename": target.name,
                    "canonical": str(canonical),
                    "target": str(target),
                    "action": f"error: {e}",
                })

    analysis_dir = canonical_results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / "canonical_png_sync_across_roots_report.csv"
    pd.DataFrame(rows, columns=["peer_root", "filename", "canonical", "target", "action"]).to_csv(report_path, index=False)
    print(f"[SYNC] Across roots: overwrote {overwritten} PNGs")
    print(f"[SYNC] Across roots report: {report_path}")
    return report_path


def cleanup_stale_generic_pngs(results_dir: Path) -> Path:
    """Delete stale generic training-results artifacts for static-only experiments."""
    viz_dir = results_dir / "visualizations"
    rows = []
    if viz_dir.exists():
        for p in viz_dir.glob("*training_results*"):
            name_lc = p.name.lower()
            delete_due_to_static_only = any(name_lc.startswith(prefix) for prefix in STALE_GENERIC_PREFIXES)
            delete_due_to_unknown_group = not any(name_lc.startswith(prefix) for prefix in ALLOWED_GENERIC_PREFIXES)
            if not delete_due_to_static_only and not delete_due_to_unknown_group:
                continue
            try:
                p.unlink()
                reason = "static_only" if delete_due_to_static_only else "unknown_group"
                rows.append({"path": str(p), "action": f"deleted::{reason}"})
            except Exception as e:
                rows.append({"path": str(p), "action": f"error: {e}"})

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / "stale_generic_cleanup_report.csv"
    pd.DataFrame(rows, columns=["path", "action"]).to_csv(report_path, index=False)
    print(f"[CLEAN] Removed {sum(str(r['action']).startswith('deleted') for r in rows)} stale generic files")
    print(f"[CLEAN] Report: {report_path}")
    return report_path


def audit_csv_quality(results_dir: Path) -> Tuple[Path, Path]:
    """Audit CSV quality (NaN/Inf/schema) across experiments for proposal traceability."""
    records = []
    for csv_path in results_dir.rglob("*.csv"):
        if not csv_path.is_file():
            continue
        if not _is_experiment_csv(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            records.append({
                "path": str(csv_path),
                "experiment": _extract_experiment_name(csv_path),
                "rows": 0,
                "status": f"read_error: {e}",
                "nan_cells": np.nan,
                "inf_cells": np.nan,
                "has_axis": False,
                "has_loss_like": False,
            })
            continue

        num = df.select_dtypes(include=[np.number])
        # Ignore expected nullable metadata columns (for example: empty error message column).
        na_view = df.copy()
        for col in list(na_view.columns):
            if str(col).strip().lower() in ALL_NULL_ALLOWLIST:
                na_view[col] = na_view[col].fillna("")
        nan_cells = int(na_view.isna().sum().sum())
        inf_cells = int(np.isinf(num.to_numpy(dtype=float)).sum()) if not num.empty else 0
        max_abs_numeric = float(np.nanmax(np.abs(num.to_numpy(dtype=float)))) if not num.empty else 0.0
        cols = [str(c).lower() for c in df.columns]
        has_axis = any(("epoch" in c) or ("iter" in c) for c in cols)
        has_loss_like = any(("loss" in c) or ("acc" in c) or ("accuracy" in c) for c in cols)
        experiment_name = _extract_experiment_name(csv_path)
        is_meta_csv = _is_metadata_csv(csv_path, experiment_name)

        status = "ok"
        if df.empty:
            status = "empty"
        elif inf_cells > 0:
            status = "has_inf"
        elif max_abs_numeric > 1e12:
            # Extremely large magnitudes often indicate numerical blow-up/divergence.
            status = "extreme_magnitude"
        elif nan_cells > 0:
            status = "has_nan"
        elif not has_loss_like:
            status = "ok_meta" if is_meta_csv else "no_metric_cols"

        records.append({
            "path": str(csv_path),
            "experiment": experiment_name,
            "rows": int(len(df)),
            "status": status,
            "nan_cells": nan_cells,
            "inf_cells": inf_cells,
            "max_abs_numeric": max_abs_numeric,
            "has_axis": bool(has_axis),
            "has_loss_like": bool(has_loss_like),
        })

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    detail_path = analysis_dir / "csv_quality_audit.csv"
    summary_path = analysis_dir / "csv_quality_audit_summary.csv"
    non_ok_path = analysis_dir / "csv_quality_audit_non_ok.csv"
    audit_df = pd.DataFrame(records)
    audit_df.to_csv(detail_path, index=False)
    if audit_df.empty:
        pd.DataFrame(columns=["status", "count"]).to_csv(summary_path, index=False)
        pd.DataFrame(columns=list(audit_df.columns)).to_csv(non_ok_path, index=False)
    else:
        audit_df.groupby("status").size().reset_index(name="count").sort_values("count", ascending=False).to_csv(
            summary_path, index=False
        )
        non_ok_df = audit_df[~audit_df["status"].isin({"ok", "ok_meta"})].copy()
        non_ok_df.to_csv(non_ok_path, index=False)
    print(f"[QC] CSV audit detail: {detail_path}")
    print(f"[QC] CSV audit summary: {summary_path}")
    print(f"[QC] CSV audit non-ok: {non_ok_path}")
    return detail_path, summary_path


def _iter_png_candidates(results_dir: Path) -> Iterable[Path]:
    for p in _iter_png_files_anycase(results_dir):
        parts_lc = [part.lower() for part in p.parts]
        if "png_contact_sheets" in parts_lc:
            continue
        yield p


def audit_png_quality(results_dir: Path) -> Tuple[Path, Path]:
    """
    Audit PNG quality and flag suspicious images (blank/degenerate/tiny/corrupt).
    """
    rows = []
    for png_path in _iter_png_candidates(results_dir):
        try:
            size_kb = png_path.stat().st_size / 1024.0
            img = mpimg.imread(png_path)
            arr = np.asarray(img)
            if arr.ndim == 2:
                gray = arr.astype(float)
            elif arr.ndim == 3:
                # Ignore alpha channel if present
                gray = arr[..., :3].mean(axis=2).astype(float)
            else:
                rows.append({
                    "path": str(png_path),
                    "status": "invalid_shape",
                    "width": np.nan,
                    "height": np.nan,
                    "size_kb": round(size_kb, 2),
                    "gray_std": np.nan,
                    "gray_range": np.nan,
                    "flag_small": True,
                    "flag_blank_like": True,
                    "flag_aspect": True,
                })
                continue

            h, w = gray.shape[:2]
            gray_std = float(np.nanstd(gray))
            gray_min = float(np.nanmin(gray))
            gray_max = float(np.nanmax(gray))
            gray_range = gray_max - gray_min
            aspect = float(w) / max(float(h), 1.0)

            flag_small = (w < 640) or (h < 360) or (size_kb < 30.0)
            flag_blank_like = (gray_std < 0.01) or (gray_range < 0.05)
            flag_aspect = (aspect > 4.5) or (aspect < 0.4)

            if flag_blank_like and flag_small:
                status = "suspicious_blank_or_tiny"
            elif flag_blank_like:
                status = "suspicious_blank_like"
            elif flag_small:
                status = "suspicious_small"
            elif flag_aspect:
                status = "suspicious_aspect"
            else:
                status = "ok"

            rows.append({
                "path": str(png_path),
                "status": status,
                "width": int(w),
                "height": int(h),
                "size_kb": round(size_kb, 2),
                "gray_std": gray_std,
                "gray_range": gray_range,
                "flag_small": bool(flag_small),
                "flag_blank_like": bool(flag_blank_like),
                "flag_aspect": bool(flag_aspect),
            })
        except Exception as e:
            rows.append({
                "path": str(png_path),
                "status": f"read_error: {e}",
                "width": np.nan,
                "height": np.nan,
                "size_kb": np.nan,
                "gray_std": np.nan,
                "gray_range": np.nan,
                "flag_small": True,
                "flag_blank_like": True,
                "flag_aspect": True,
            })

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    detail_path = analysis_dir / "png_quality_audit.csv"
    summary_path = analysis_dir / "png_quality_audit_summary.csv"

    audit_df = pd.DataFrame(rows)
    if audit_df.empty:
        pd.DataFrame(columns=["status", "count"]).to_csv(summary_path, index=False)
        audit_df.to_csv(detail_path, index=False)
    else:
        audit_df.to_csv(detail_path, index=False)
        (
            audit_df.groupby("status")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .to_csv(summary_path, index=False)
        )

    print(f"[QC] PNG audit detail: {detail_path}")
    print(f"[QC] PNG audit summary: {summary_path}")
    return detail_path, summary_path


def audit_proposal_required_artifacts(results_dir: Path) -> Tuple[Path, Path]:
    """
    Check presence of proposal-required visualization artifacts.
    """
    rows = []
    for exp_name, patterns in PROPOSAL_REQUIRED_PATTERNS.items():
        for rel in patterns:
            matches = list(results_dir.glob(rel))
            if matches:
                for m in matches:
                    rows.append({
                        "experiment": exp_name,
                        "pattern": rel,
                        "status": "present",
                        "path": str(m),
                    })
            else:
                rows.append({
                    "experiment": exp_name,
                    "pattern": rel,
                    "status": "missing",
                    "path": "",
                })

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    detail_path = analysis_dir / "proposal_required_artifacts_audit.csv"
    summary_path = analysis_dir / "proposal_required_artifacts_summary.csv"

    out_df = pd.DataFrame(rows, columns=["experiment", "pattern", "status", "path"])
    out_df.to_csv(detail_path, index=False)
    (
        out_df.groupby(["experiment", "status"])
        .size()
        .reset_index(name="count")
        .sort_values(["experiment", "status"])
        .to_csv(summary_path, index=False)
    )
    print(f"[QC] Proposal artifact audit detail: {detail_path}")
    print(f"[QC] Proposal artifact audit summary: {summary_path}")
    return detail_path, summary_path


def write_proposal_expectation_mapping(results_dir: Path) -> Tuple[Path, Path]:
    """
    Extract proposal-required visualization/metric expectations and map them to current generators.
    """
    generator_map = {
        "2d_optimization": "run_all_kaggle.py::_create_2d_visualizations",
        "robustness": "run_all_kaggle.py::_create_robustness_visualizations",
        "sam_sensitivity": "run_all_kaggle.py::_create_sam_sensitivity_visualizations",
        "ablation": "run_all_kaggle.py::_create_ablation_visualizations",
        "beta_sensitivity_2d": "run_beta_2d_demos.py::main / generate_*",
    }

    rows = []
    for exp_name, patterns in PROPOSAL_REQUIRED_PATTERNS.items():
        metric_cfg = PROPOSAL_REQUIRED_METRICS.get(exp_name, {})
        for rel in patterns:
            rows.append(
                {
                    "experiment": exp_name,
                    "expectation_type": "visualization",
                    "expected_artifact": rel,
                    "generator": generator_map.get(exp_name, "unknown"),
                    "source_docs": "; ".join(PROPOSAL_SOURCE_DOCS),
                }
            )
        if metric_cfg:
            rows.append(
                {
                    "experiment": exp_name,
                    "expectation_type": "metrics_schema",
                    "expected_artifact": metric_cfg.get("summary_csv", ""),
                    "generator": generator_map.get(exp_name, "unknown"),
                    "source_docs": "; ".join(PROPOSAL_SOURCE_DOCS),
                }
            )

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "proposal_expectations_mapping.csv"
    md_path = analysis_dir / "proposal_expectations_mapping.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Proposal Expectations Mapping\n\n")
        f.write(f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        f.write(f"- Results root: `{results_dir}`\n")
        f.write("- Source docs:\n")
        for src in PROPOSAL_SOURCE_DOCS:
            f.write(f"  - `{src}`\n")
        f.write("\n## Convergence/Metric Rules Applied\n")
        for note in PROPOSAL_EXPECTATION_NOTES:
            f.write(f"- {note}\n")
        f.write("\n## Experiment Mapping\n")
        for exp_name in sorted(PROPOSAL_REQUIRED_PATTERNS.keys()):
            f.write(f"\n### {exp_name}\n")
            f.write(f"- Generator: `{generator_map.get(exp_name, 'unknown')}`\n")
            metric_cfg = PROPOSAL_REQUIRED_METRICS.get(exp_name, {})
            if metric_cfg:
                f.write(f"- Summary CSV: `{metric_cfg.get('summary_csv', '')}`\n")
                expected_seeds = metric_cfg.get("expected_unique_seeds")
                if expected_seeds is not None:
                    f.write(f"- Expected unique seeds: `{expected_seeds}`\n")
                req_cols = metric_cfg.get("required_columns", [])
                if req_cols:
                    f.write(f"- Required columns: `{', '.join(req_cols)}`\n")
            f.write("- Required visualization artifacts:\n")
            for rel in PROPOSAL_REQUIRED_PATTERNS.get(exp_name, []):
                f.write(f"  - `{rel}`\n")

    print(f"[MAP] Proposal expectation mapping CSV: {csv_path}")
    print(f"[MAP] Proposal expectation mapping MD: {md_path}")
    return csv_path, md_path


def _non_empty_count(series: pd.Series) -> int:
    if pd.api.types.is_numeric_dtype(series):
        return int(pd.to_numeric(series, errors="coerce").notna().sum())
    txt = series.astype(str).str.strip().str.lower()
    return int((~txt.isin({"", "nan", "none", "null"})).sum())


def audit_proposal_metric_expectations(results_dir: Path) -> Tuple[Path, Path]:
    """
    Audit proposal-critical metrics/schema expectations in canonical summary CSVs.
    """
    rows = []
    for exp_name, cfg in PROPOSAL_REQUIRED_METRICS.items():
        rel_csv = str(cfg.get("summary_csv", ""))
        expected_unique_seeds = int(cfg.get("expected_unique_seeds", 0) or 0)
        csv_path = results_dir / rel_csv
        req_cols = list(cfg.get("required_columns", []))
        metric_alts = list(cfg.get("at_least_one_metric_cols", []))
        conv_alts = list(cfg.get("at_least_one_convergence_cols", []))

        if not csv_path.exists():
            rows.append(
                {
                    "experiment": exp_name,
                    "summary_csv": rel_csv,
                    "status": "missing_csv",
                    "rows": 0,
                    "n_seeds": 0,
                    "expected_unique_seeds": expected_unique_seeds,
                    "seed_shortfall": expected_unique_seeds,
                    "missing_required_columns": "|".join(req_cols),
                    "missing_metric_alt": bool(metric_alts),
                    "missing_convergence_alt": bool(conv_alts),
                    "all_null_columns": "",
                    "error_rows": 0,
                    "nonfinite_final_loss_rows": 0,
                }
            )
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            rows.append(
                {
                    "experiment": exp_name,
                    "summary_csv": rel_csv,
                    "status": f"read_error: {e}",
                    "rows": 0,
                    "n_seeds": 0,
                    "expected_unique_seeds": expected_unique_seeds,
                    "seed_shortfall": expected_unique_seeds,
                    "missing_required_columns": "|".join(req_cols),
                    "missing_metric_alt": bool(metric_alts),
                    "missing_convergence_alt": bool(conv_alts),
                    "all_null_columns": "",
                    "error_rows": 0,
                    "nonfinite_final_loss_rows": 0,
                }
            )
            continue

        if df.empty:
            rows.append(
                {
                    "experiment": exp_name,
                    "summary_csv": rel_csv,
                    "status": "empty_csv",
                    "rows": 0,
                    "n_seeds": 0,
                    "expected_unique_seeds": expected_unique_seeds,
                    "seed_shortfall": expected_unique_seeds,
                    "missing_required_columns": "|".join([c for c in req_cols if c not in df.columns]),
                    "missing_metric_alt": (bool(metric_alts) and not any(c in df.columns for c in metric_alts)),
                    "missing_convergence_alt": (bool(conv_alts) and not any(c in df.columns for c in conv_alts)),
                    "all_null_columns": "",
                    "error_rows": 0,
                    "nonfinite_final_loss_rows": 0,
                }
            )
            continue

        missing_required = [c for c in req_cols if c not in df.columns]
        missing_metric_alt = bool(metric_alts) and not any(c in df.columns for c in metric_alts)
        missing_convergence_alt = bool(conv_alts) and not any(c in df.columns for c in conv_alts)

        all_null_cols = []
        for c in req_cols:
            if c in ALL_NULL_ALLOWLIST:
                continue
            if c in df.columns and _non_empty_count(df[c]) == 0:
                all_null_cols.append(c)

        n_seeds = 0
        if "seed" in df.columns:
            n_seeds = int(pd.to_numeric(df["seed"], errors="coerce").dropna().nunique())
        seed_shortfall = max(0, expected_unique_seeds - n_seeds) if expected_unique_seeds > 0 else 0

        error_mask = pd.Series(False, index=df.index)
        if "run_status" in df.columns:
            status_txt = df["run_status"].astype(str).str.strip().str.lower()
            error_mask = error_mask | status_txt.str.startswith("error")
        if "error" in df.columns:
            err_txt = df["error"].astype(str).str.strip().str.lower()
            error_mask = error_mask | (~err_txt.isin({"", "nan", "none", "null"}))
        error_rows = int(error_mask.sum())

        nonfinite_final_loss_rows = 0
        if "final_loss" in df.columns:
            final_loss = pd.to_numeric(df["final_loss"], errors="coerce").to_numpy(dtype=float)
            nonfinite_final_loss_rows = int((~np.isfinite(final_loss)).sum())

        status = "ok"
        if missing_required or missing_metric_alt or missing_convergence_alt:
            status = "schema_gap"
        elif all_null_cols:
            status = "all_null_columns"
        elif nonfinite_final_loss_rows > 0:
            status = "nonfinite_losses"
        elif error_rows > 0:
            status = "error_rows_present"
        elif seed_shortfall > 0:
            status = "seed_coverage_gap"

        rows.append(
            {
                "experiment": exp_name,
                "summary_csv": rel_csv,
                "status": status,
                "rows": int(len(df)),
                "n_seeds": n_seeds,
                "expected_unique_seeds": expected_unique_seeds,
                "seed_shortfall": seed_shortfall,
                "missing_required_columns": "|".join(missing_required),
                "missing_metric_alt": bool(missing_metric_alt),
                "missing_convergence_alt": bool(missing_convergence_alt),
                "all_null_columns": "|".join(all_null_cols),
                "error_rows": error_rows,
                "nonfinite_final_loss_rows": nonfinite_final_loss_rows,
            }
        )

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    detail_path = analysis_dir / "proposal_metric_expectations_audit.csv"
    summary_path = analysis_dir / "proposal_metric_expectations_summary.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(detail_path, index=False)
    if out_df.empty:
        pd.DataFrame(columns=["status", "count"]).to_csv(summary_path, index=False)
    else:
        (
            out_df.groupby("status")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .to_csv(summary_path, index=False)
        )
    print(f"[QC] Proposal metric audit detail: {detail_path}")
    print(f"[QC] Proposal metric audit summary: {summary_path}")
    return detail_path, summary_path


def _get_x_axis(df):
    """Robustly detect x-axis column (prioritizing epoch > iteration > index)."""
    def _sanitize_axis(raw_vals: Any, fallback_len: int) -> np.ndarray:
        arr = pd.to_numeric(pd.Series(raw_vals), errors='coerce').to_numpy(dtype=float)
        if arr.size == 0:
            return np.arange(1, fallback_len + 1, dtype=float)
        if not np.isfinite(arr).all():
            return np.arange(1, fallback_len + 1, dtype=float)
        # Guard against pathological ranges that can explode renderer/tight_bbox
        rng = float(np.nanmax(arr) - np.nanmin(arr)) if arr.size else 0.0
        if rng > 1e7:
            return np.arange(1, fallback_len + 1, dtype=float)
        return arr

    # Look for 'epoch' or anything containing it
    epoch_col = next((col for col in df.columns if col.strip().lower() == 'epoch' or 'epoch' in col.lower()), None)
    if epoch_col is not None:
        return _sanitize_axis(df[epoch_col].values, len(df)), "Epoch"
    
    # Fallback to iteration
    iter_col = next((col for col in df.columns if 'iter' in col.lower()), None)
    if iter_col is not None:
        return _sanitize_axis(df[iter_col].values, len(df)), "Iteration"
        
    # Final fallback: row index
    return np.arange(1, len(df) + 1), "Epoch"


def plot_training_curves(csv_files: List[str], output_dir: Path, title: str = "Training Curves") -> bool:
    """
    Generate training curves from CSV files.
    Handles MNIST, CIFAR-10, NLP, and other NN experiments.
    """
    def _cleanup_stale_outputs() -> None:
        base_name = title.lower().replace(' ', '_')
        stale_paths = [
            output_dir / f"{base_name}.png",
            output_dir / f"{base_name}_summary.csv",
            output_dir / f"{base_name}_summary.md",
        ]
        removed = 0
        for path in stale_paths:
            if path.exists():
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    continue
        if removed:
            print(f"[CLEAN] Removed {removed} stale outputs for {title}")

    if not csv_files:
        _cleanup_stale_outputs()
        return False

    # Patterns that indicate meta/aggregator CSV files (not actual per-epoch training runs).
    # Any file starting with 'advablation_' is a combined ablation summary - skip it.

    def _canonicalize_optimizer_name(name: str, exp_name: Optional[str]) -> str:
        cleaned = str(name)
        # Remove chained run suffixes such as _start0_seed42 or _trial3_run2
        cleaned = re.sub(r'(?:_(?:start|trial|seed|run)\d+)+$', '', cleaned, flags=re.IGNORECASE)
        # Remove any remaining start markers if embedded
        cleaned = re.sub(r'_start\d+', '', cleaned, flags=re.IGNORECASE)

        exp_lc = (exp_name or '').lower()
        if exp_lc in {'robustness', 'adversarial'}:
            # Collapse aliases such as adversarial__from_robustness__...
            cleaned = re.sub(r'(?i)^adversarial__from_robustness__', '', cleaned)
            # Robustness files usually include function/startpoint prefixes; keep only optimizer family
            m = re.search(r'(?i)Robustness_[^_]+_(.+)$', cleaned)
            if m:
                cleaned = m.group(1)

        cleaned = cleaned.strip('_')
        cleaned = re.sub(r'__+', '_', cleaned)
        return cleaned

    # Group by optimizer
    results = {}
    experiment_name = None
    if csv_files:
        experiment_name = _extract_experiment_name(Path(csv_files[0]))
    min_len = 5
    if experiment_name is not None and experiment_name.lower() in {"medical", "batch_ablation"}:
        min_len = 1
    for csv_file in csv_files:
        path_obj = Path(csv_file)
        basename = path_obj.name.lower()

        if _is_meta_csv(path_obj) and not any(run_pat in basename for run_pat in ["_seed", "_start", "_trial", "_run"]):
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue

        if not _looks_like_training_timeseries(df, min_len=min_len):
            continue

        # Extract label from filename (removes trial/seed suffixes)
        base = os.path.basename(csv_file).replace('.csv', '')
        optimizer = _canonicalize_optimizer_name(base, experiment_name)
        
        # Clean up known redundant dataset prefixes
        prefixes_to_strip = ['MNIST_', 'CIFAR10_', 'IMDB_', 'MEDICAL_', 'AdvAblation_', '2D_', 'Analysis_']
        for p in prefixes_to_strip:
            if optimizer.lower().startswith(p.lower()):
                optimizer = optimizer[len(p):]
                break
                
        # Additional formatting: sometimes names start with underscores or are just entirely redundant
        if optimizer.startswith('_'): optimizer = optimizer[1:]
        
        # Fallback if empty (e.g. filename was just the prefix)
        if not optimizer:
            if 'optimizer' in df.columns:
                opt_val = df['optimizer'].dropna()
                if not opt_val.empty:
                    v = str(opt_val.iloc[0])
                    if v not in ('Unknown', 'nan', 'MISSING', ''):
                        optimizer = v

        # Skip files we still can't identify - they are noise, not real runs
        if not optimizer or str(optimizer).lower().strip() in ('unknown', 'nan', '', 'missing', 'none'):
            continue

        if optimizer not in results:
            results[optimizer] = []
        results[optimizer].append(df)

    if not results:
        print(f"[SKIP] {title}: no valid training time-series CSV files")
        _cleanup_stale_outputs()
        return False

    # Normalize and sort optimizers for stable color assignment
    sorted_optimizers = sorted(results.keys())
    total_runs = int(sum(len(v) for v in results.values()))
    heavy_plot = (len(sorted_optimizers) >= 25) or (total_runs >= 250)
    
    # Define stable colors
    base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FFD93D', '#B8E6F1', 
                   '#A29BFE', '#FAB1A0', '#55E6C1', '#25CCF7', '#FD7272', '#58B19F', '#BDC581']
    opt_to_color = {opt: base_colors[i % len(base_colors)] for i, opt in enumerate(sorted_optimizers)}

    # DEBUG: trace what landed in results
    print(f'[DEBUG] plot_training_curves({title!r}): optimizers={sorted_optimizers!r}', file=sys.stderr)

    # Create figure - adapt to workload to avoid renderer OOM on very large groups
    fig_size = (16, 10) if heavy_plot else (24, 16)
    save_dpi = 180 if heavy_plot else 300
    draw_individual_runs = not heavy_plot
    max_points = 2000 if heavy_plot else None

    def _downsample_xy(x_vals: np.ndarray, y_vals: np.ndarray, max_n: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        if max_n is None:
            return x_vals, y_vals
        n = min(len(x_vals), len(y_vals))
        if n <= max_n:
            return x_vals[:n], y_vals[:n]
        step = max(1, int(np.ceil(n / max_n)))
        return x_vals[:n:step], y_vals[:n:step]

    def _pick_metric_column(df: pd.DataFrame, want_test: bool = True) -> Optional[str]:
        cols = [str(c) for c in df.columns]
        if want_test:
            candidates = [c for c in cols if ('test' in c.lower()) and any(t in c.lower() for t in ['acc', 'accuracy', 'dice'])]
            if candidates:
                return candidates[0]
        val_candidates = [c for c in cols if ('val' in c.lower()) and any(t in c.lower() for t in ['acc', 'accuracy', 'dice'])]
        if val_candidates:
            return val_candidates[0]
        train_candidates = [c for c in cols if ('train' in c.lower()) and any(t in c.lower() for t in ['acc', 'accuracy', 'dice'])]
        if train_candidates:
            return train_candidates[0]
        return None

    # Pre-scan available panels to avoid rendering empty subplot slots
    has_test_acc_panel = False
    for _, dfs in sorted(results.items()):
        for df in dfs:
            acc_col = _pick_metric_column(df, want_test=True)
            if not acc_col:
                continue
            acc_vals = pd.to_numeric(df[acc_col], errors='coerce')
            if acc_vals.isna().all() or (acc_vals == 0.01).all() or (acc_vals == 0).all() or (acc_vals == 0.5).all():
                continue
            has_test_acc_panel = True
            break
        if has_test_acc_panel:
            break

    final_metrics = {}
    final_stds = {}
    final_metric_label = 'Final Metric (%)'
    final_metric_title = 'Final Performance'
    final_metric_is_loss = False
    for optimizer, dfs in results.items():
        final_vals = []
        for df in dfs:
            acc_col = _pick_metric_column(df, want_test=True)
            if acc_col:
                val = df[acc_col].iloc[-1]
                if val <= 1.0:
                    val = val * 100
                final_vals.append(val)
        # Skip degenerate placeholder-style "accuracy" (e.g., all 0.01 -> 1.00%)
        if final_vals and not (
            len(final_vals) > 0
            and np.nanstd(final_vals) < 1e-9
            and np.isclose(np.nanmean(final_vals), 1.0, atol=1e-6)
        ):
            final_metrics[optimizer] = np.mean(final_vals)
            final_stds[optimizer] = np.std(final_vals) if len(final_vals) > 1 else 0

    # Fallback: if no valid accuracy metric, use final loss by optimizer (lower is better)
    if not final_metrics:
        loss_means = {}
        loss_stds = {}
        for optimizer, dfs in results.items():
            final_losses = []
            for df in dfs:
                loss_col = next((col for col in df.columns if 'train_loss' == col.lower()), None)
                if not loss_col:
                    loss_col = next((col for col in df.columns if 'loss' in col.lower() and 'test' not in col.lower() and 'val' not in col.lower()), None)
                if not loss_col:
                    loss_col = next((col for col in df.columns if 'loss' in col.lower()), None)
                if loss_col and len(df[loss_col]) > 0:
                    val = pd.to_numeric(pd.Series([df[loss_col].iloc[-1]]), errors='coerce').iloc[0]
                    if pd.notna(val):
                        final_losses.append(float(val))
            if final_losses:
                loss_means[optimizer] = float(np.mean(final_losses))
                loss_stds[optimizer] = float(np.std(final_losses)) if len(final_losses) > 1 else 0.0
        if loss_means:
            final_metrics = loss_means
            final_stds = loss_stds
            final_metric_label = 'Final Loss (lower is better)'
            final_metric_title = 'Final Loss'
            final_metric_is_loss = True

    final_metrics = {k: v for k, v in final_metrics.items() if str(k).lower() not in ('unknown', 'nan', '', 'missing')}
    has_final_performance_panel = len(final_metrics) > 0

    speeds = {}
    for optimizer, dfs in results.items():
        run_speeds = []
        for df in dfs:
            elapsed = None
            if 'elapsed_seconds' in df.columns and df['elapsed_seconds'].max() > 0:
                elapsed = df['elapsed_seconds'].max()
            elif 'time_sec' in df.columns:
                max_time = df['time_sec'].max()
                if max_time > 0:
                    elapsed = max_time
            elif 'time' in df.columns:
                max_time = df['time'].max()
                if max_time > 0:
                    elapsed = max_time

            if elapsed is not None:
                if abs(elapsed - len(df)) < 1e-3:
                    continue
                if (
                    elapsed % 1 == 0
                    and len(df) % 10 == 0
                    and 'elapsed_seconds' in df.columns
                    and df['elapsed_seconds'].nunique() <= 2
                ):
                    continue
                x_vals, _ = _get_x_axis(df)
                if len(x_vals) > 0 and elapsed > 1e-4:
                    max_x = np.max(x_vals)
                    speed = max_x / elapsed
                    run_speeds.append(speed)
        if run_speeds:
            speeds[optimizer] = np.mean(run_speeds)

    has_speed_panel = False
    if speeds and len(speeds) > 0:
        vals = list(speeds.values())
        synthetic_only = ((max(vals) - min(vals) < 1e-5) and np.isclose(max(vals), 1000.0, atol=1e-3))
        has_speed_panel = not synthetic_only

    panel_specs = [('loss', 'Training Loss Curves')]
    if has_test_acc_panel:
        panel_specs.append(('accuracy', 'Accuracy'))
    if has_final_performance_panel:
        panel_specs.append(('final', 'Final Performance'))
    if has_speed_panel:
        panel_specs.append(('speed', 'Training Efficiency'))

    n_panels = len(panel_specs)
    if n_panels == 3:
        ncols = 3
    elif n_panels > 1:
        ncols = 2
    else:
        ncols = 1
    nrows = int(np.ceil(n_panels / ncols))
    width_scale = max(1.0, ncols / 2)
    height_scale = max(0.75, nrows / 2)
    dynamic_fig_size = (
        int(np.ceil(fig_size[0] * width_scale)),
        max(6, int(np.ceil(fig_size[1] * height_scale))),
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=dynamic_fig_size)
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    axes_flat = np.atleast_1d(axes).ravel()
    panel_axes = {name: axes_flat[idx] for idx, (name, _) in enumerate(panel_specs)}
    for extra_ax in axes_flat[n_panels:]:
        extra_ax.set_visible(False)

    # Plot 1: Training Loss
    ax = panel_axes['loss']
    x_label_for_plot = "Epoch / Iteration" # Initialize for category
    for i, (optimizer, dfs) in enumerate(sorted(results.items())):
        color = opt_to_color[optimizer]
        
        # Collect per-run (x_vals, loss_vals) for mean calculation
        runs_for_mean = []
        # (local updates below)
        for df in dfs:
            x_vals, x_label = _get_x_axis(df)
            x_label_for_plot = x_label # Use the label from the first valid run

            # Identify loss column (prioritize train_loss, then generic loss, exclude test/val)
            loss_col = next((col for col in df.columns if 'train_loss' == col.lower()), None)
            if not loss_col:
                loss_col = next((col for col in df.columns if 'loss' in col.lower() and 'test' not in col.lower() and 'val' not in col.lower()), None)
            if not loss_col: # Fallback to any loss column if specific not found
                loss_col = next((col for col in df.columns if 'loss' in col.lower()), None)
                
            if loss_col and loss_col in df.columns:
                # Plot individual run after grouping by x_axis
                run_df = pd.DataFrame({'x': x_vals, 'y': df[loss_col]}).groupby('x').mean().reset_index()
                if draw_individual_runs:
                    xs = run_df['x'].to_numpy()
                    ys = pd.to_numeric(run_df['y'], errors='coerce').to_numpy(dtype=float)
                    xs, ys = _downsample_xy(xs, ys, max_points)
                    ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1)
                runs_for_mean.append((run_df['x'].values, run_df['y'].values))

        # Mean line (align runs by common x-axis grid using interpolation)
        if runs_for_mean:
            # Determine common x-axis range, ensuring we have valid finite bounds
            valid_runs = [(x, y) for x, y in runs_for_mean if len(x) > 0 and np.isfinite(x).all()]
            if not valid_runs:
                continue
                
            min_x = int(min(e.min() for e, _ in valid_runs))
            max_x = int(max(e.max() for e, _ in valid_runs))

            # Sanity check on bounds
            if max_x < min_x or max_x - min_x > 100000:  # Prevent massive arange calls
                continue

            common_x = np.arange(min_x, max_x + 1)

            aligned_losses = []
            for x_run, loss_run in valid_runs:
                s = pd.Series(loss_run, index=x_run)
                if not s.index.is_unique:
                    s = s.groupby(level=0).mean()
                # Use interpolation but handle boundaries safely
                try:
                    s = s.reindex(common_x).interpolate(method='linear').ffill().bfill().values
                    if np.isfinite(s).all():
                        aligned_losses.append(s)
                except Exception:
                    continue
            
            if aligned_losses:
                mean_loss = np.mean(np.vstack(aligned_losses), axis=0)
                dx, dy = _downsample_xy(np.asarray(common_x), np.asarray(mean_loss), max_points)
                ax.plot(dx, dy, color=color, linewidth=2.5, label=optimizer)
                if len(dx) == 1:
                    ax.scatter(dx, dy, color=color, s=40, zorder=3)

    ax.set_xlabel(x_label_for_plot, fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss Curves', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    # Move legend outside the plot area
    if len(sorted_optimizers) <= 20:
        ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    else:
        ax.text(0.98, 0.02, f'Legend omitted ({len(sorted_optimizers)} series)', ha='right', va='bottom', transform=ax.transAxes, fontsize=9)

    # Plot 2: Test/Validation Accuracy (only when available)
    if has_test_acc_panel:
        ax = panel_axes['accuracy']
        metric_label = "Test Accuracy (%)"
        x_label_for_plot = "Epoch"
        for i, (optimizer, dfs) in enumerate(sorted(results.items())):
            color = opt_to_color[optimizer]

            runs_for_mean = []

            for df in dfs:
                x_vals, x_label = _get_x_axis(df)
                x_label_for_plot = x_label

                acc_col = _pick_metric_column(df, want_test=True)

                if acc_col:
                    acc_vals = pd.to_numeric(df[acc_col], errors='coerce')
                    if acc_vals.isna().all() or (acc_vals == 0.01).all() or (acc_vals == 0).all() or (acc_vals == 0.5).all():
                        continue

                    if acc_vals.max() <= 1.01:
                        acc_vals = acc_vals * 100.0

                    if acc_col and 'dice' in acc_col.lower():
                        metric_label = "Test/Val Dice (%)"
                    elif acc_col and 'val' in acc_col.lower():
                        metric_label = "Validation Accuracy (%)"
                    else:
                        metric_label = "Test Accuracy (%)"

                    run_df = pd.DataFrame({'x': x_vals, 'y': acc_vals}).groupby('x').mean().reset_index()
                    if draw_individual_runs:
                        xs = run_df['x'].to_numpy()
                        ys = pd.to_numeric(run_df['y'], errors='coerce').to_numpy(dtype=float)
                        xs, ys = _downsample_xy(xs, ys, max_points)
                        ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1)
                    runs_for_mean.append((run_df['x'].values, run_df['y'].values))

            if runs_for_mean:
                valid_runs = [(x, y) for x, y in runs_for_mean if len(x) > 0 and np.isfinite(x).all()]
                if not valid_runs:
                    continue

                min_x = int(min(e.min() for e, _ in valid_runs))
                max_x = int(max(e.max() for e, _ in valid_runs))

                if max_x < min_x or max_x - min_x > 100000:
                    continue

                common_x = np.arange(min_x, max_x + 1)

                aligned_accs = []
                for e, a in valid_runs:
                    s = pd.Series(a, index=e)
                    if not s.index.is_unique:
                        s = s.groupby(level=0).mean()
                    try:
                        s = s.reindex(common_x).interpolate().ffill().bfill().values
                        if np.isfinite(s).all():
                            aligned_accs.append(s)
                    except Exception:
                        continue

                if aligned_accs:
                    mean_acc = np.mean(np.vstack(aligned_accs), axis=0)
                    dx, dy = _downsample_xy(np.asarray(common_x), np.asarray(mean_acc), max_points)
                    ax.plot(dx, dy, color=color, linewidth=2.5, label=optimizer)

        ax.set_xlabel(x_label_for_plot, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
        ax.set_title(metric_label.replace(' (%)', ''), fontsize=16, fontweight='bold', pad=15)
        if len(sorted_optimizers) <= 20:
            ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        else:
            ax.text(0.98, 0.02, f'Legend omitted ({len(sorted_optimizers)} series)', ha='right', va='bottom', transform=ax.transAxes, fontsize=9)
        ax.grid(True, alpha=0.3)

    # Plot 3: Final Performance Bar Chart (only when available)
    if has_final_performance_panel:
        ax = panel_axes['final']
        optimizers_sorted = sorted(final_metrics.keys(), key=lambda k: final_metrics[k], reverse=(not final_metric_is_loss))
        max_bars = 15
        if len(optimizers_sorted) > max_bars:
            optimizers_plot = optimizers_sorted[:max_bars]
        else:
            optimizers_plot = optimizers_sorted

        x_pos = np.arange(len(optimizers_plot))
        bars = ax.bar(x_pos, [final_metrics[opt] for opt in optimizers_plot],
                      yerr=[final_stds[opt] for opt in optimizers_plot],
                      color=[opt_to_color[opt] for opt in optimizers_plot],
                      alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizers_plot, rotation=35, ha='right', fontsize=10)
        ax.set_ylabel(final_metric_label, fontsize=14, fontweight='bold')
        ax.set_title(final_metric_title, fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)

        if len(optimizers_plot) <= 12:
            panel_max = max([abs(final_metrics[o]) for o in optimizers_plot]) if optimizers_plot else 1.0
            offset = 0.01 * max(1.0, panel_max)
            for bar, opt in zip(bars, optimizers_plot):
                height = bar.get_height()
                if (not np.isfinite(height)) or abs(height) > 1e6:
                    continue
                label_text = f'{final_metrics[opt]:.2f}' + ('' if final_metric_is_loss else '%')
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                        label_text,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        if len(optimizers_sorted) > max_bars:
            ax.text(0.98, 0.98, f'Top {max_bars} of {len(optimizers_sorted)}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)

    # Plot 4: Training speed comparison (only when available)
    if has_speed_panel:
        ax = panel_axes['speed']
        sorted_keys = sorted(speeds.keys(), key=lambda k: speeds[k], reverse=True)
        max_bars = 15
        if len(sorted_keys) > max_bars:
            sorted_keys_plot = sorted_keys[:max_bars]
        else:
            sorted_keys_plot = sorted_keys

        x_pos = np.arange(len(sorted_keys_plot))
        bars = ax.bar(x_pos, [speeds[k] for k in sorted_keys_plot],
                      color=[opt_to_color[k] for k in sorted_keys_plot],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_keys_plot, rotation=35, ha='right', fontsize=10)
        ax.set_ylabel('Speed (Epochs or Steps / Sec)', fontsize=14, fontweight='bold')
        ax.set_title('Training Efficiency', fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        if len(sorted_keys) > max_bars:
            ax.text(0.98, 0.98, f'Top {max_bars} of {len(sorted_keys)}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / f"{title.lower().replace(' ', '_')}.png"
    try:
        plt.savefig(output_file, dpi=save_dpi, bbox_inches='tight')
    except ValueError as e:
        # Fallback for pathological extents from outlier values/text artists
        print(f"[WARN] savefig tight bbox failed for {title}: {e}. Retrying with safe renderer settings.")
        try:
            plt.savefig(output_file, dpi=max(120, int(save_dpi * 0.67)))
        except Exception as e2:
            print(f"[ERROR] savefig fallback failed for {title}: {e2}")
            raise
    print(f"[OK] Saved: {output_file}")
    plt.close()

    # Generate Tabular Summary
    try:
        summary_data = []
        for optimizer, dfs in results.items():
            if final_metric_is_loss:
                acc = np.nan
                acc_std = np.nan
            else:
                acc = final_metrics.get(optimizer, np.nan)
                acc_std = final_stds.get(optimizer, np.nan)
            speed = speeds.get(optimizer, np.nan) if has_speed_panel else np.nan
            
            final_losses = []
            for df in dfs:
                loss_col = next((col for col in df.columns if 'train_loss' == col.lower()), None)
                if not loss_col:
                    loss_col = next((col for col in df.columns if 'loss' in col.lower() and 'test' not in col.lower() and 'val' not in col.lower()), None)
                if not loss_col:
                    loss_col = next((col for col in df.columns if 'loss' in col.lower()), None)
                if loss_col and len(df[loss_col]) > 0:
                    final_losses.append(df[loss_col].iloc[-1])
            
            loss_val = np.mean(final_losses) if final_losses else np.nan
            loss_std = np.std(final_losses) if len(final_losses) > 1 else np.nan
            
            if str(optimizer).lower().strip() not in ('unknown', 'nan', '', 'missing'):
                summary_data.append({
                    'Optimizer/Config': optimizer,
                    'Final Loss': loss_val,
                    'Loss Std': loss_std,
                    'Final Test Acc (%)': acc,
                    'Acc Std (%)': acc_std,
                    'Speed (iters/sec)': speed
                })
            
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Sort by best accuracy, or lowest loss if accuracy isn't available
            if summary_df['Final Test Acc (%)'].notna().any():
                summary_df = summary_df.sort_values(by='Final Test Acc (%)', ascending=False)
            else:
                summary_df = summary_df.sort_values(by='Final Loss', ascending=True)
                
            # Formatting
            for col in ['Final Loss', 'Loss Std', 'Final Test Acc (%)', 'Acc Std (%)', 'Speed (iters/sec)']:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
            csv_file = output_dir / f"{title.lower().replace(' ', '_')}_summary.csv"
            summary_df.to_csv(csv_file, index=False)
            
            md_file = output_dir / f"{title.lower().replace(' ', '_')}_summary.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(f"## {title} - Tabular Summary\n\n")
                
                # Manual markdown table generation to avoid 'tabulate' dependency
                headers = summary_df.columns.tolist()
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|\n")
                for _, row in summary_df.iterrows():
                    f.write("| " + " | ".join(str(x) for x in row) + " |\n")
                    
            print(f"[OK] Saved table: {md_file}")
    except Exception as e:
        print(f"[WARN] Failed to generate table for {title}: {e}")
    return True


def generate_all_plots(
    results_dir: str = 'results',
    include_static_only: bool = False,
    overwrite_duplicates: bool = False,
    sync_peer_dirs: Optional[List[str]] = None,
    cleanup_stale: bool = True,
    run_csv_audit: bool = True,
    run_png_audit: bool = True,
    run_proposal_artifact_audit: bool = True,
    run_proposal_metric_audit: bool = True,
    write_proposal_map: bool = True,
):
    """
    Automatically generate plots for all experiments in results directory.
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    print(f"[PLOTS] Generating visualizations from: {results_dir}")
    print("="*80)

    if overwrite_duplicates:
        overwrite_duplicate_pngs(results_path)
    if sync_peer_dirs:
        peer_paths = [Path(p) for p in sync_peer_dirs if str(p).strip()]
        if peer_paths:
            overwrite_duplicate_pngs_across_roots(results_path, peer_paths)
    if cleanup_stale and not include_static_only:
        cleanup_stale_generic_pngs(results_path)
    backfill_convergence_columns(results_path)
    _maybe_generate_theory_practice_status_plot(results_path)

    # Find all CSV files under experiments only (proposal-aligned scope)
    all_csvs = [p for p in results_path.rglob("*.csv") if p.is_file() and _is_experiment_csv(p)]

    if not all_csvs:
        print("[WARN] No CSV files found")
        return

    # Group by experiment type dynamically
    experiments: Dict[str, List[str]] = {}
    skipped_counts: Dict[str, int] = {}

    for csv_file in all_csvs:
        exp_name = _extract_experiment_name(csv_file)
        if not exp_name:
            skipped_counts["missing_experiment_name"] = skipped_counts.get("missing_experiment_name", 0) + 1
            continue

        exp_name_lc = exp_name.lower()
        if (not include_static_only) and exp_name_lc in STATIC_ONLY_EXPERIMENTS:
            skipped_counts[f"static_only::{exp_name_lc}"] = skipped_counts.get(f"static_only::{exp_name_lc}", 0) + 1
            continue
        if _is_meta_csv(csv_file):
            skipped_counts["meta_csv"] = skipped_counts.get("meta_csv", 0) + 1
            continue

        category = exp_name.replace('_', ' ').title()
        experiments.setdefault(category, []).append(str(csv_file))

    # Generate plots for each category
    viz_dir = results_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    plots_created = 0
    groups_seen = 0
    for exp_type, csv_files in experiments.items():
        if csv_files:
            groups_seen += 1
            print(f"\n[GROUP] {exp_type}: {len(csv_files)} files")
            try:
                ok = plot_training_curves(csv_files, viz_dir, title=f"{exp_type} Training Results")
                if ok:
                    plots_created += 1
            except Exception as e:
                print(f"   [WARN] Error in {exp_type}: {e}")
                traceback.print_exc()

    print("\n" + "="*80)
    print(f"[OK] Created {plots_created} visualization sets (from {groups_seen} groups) in: {viz_dir}")
    print(f"   All plots are high-quality (300 DPI)")
    if skipped_counts:
        qc_df = pd.DataFrame([
            {"reason": k, "count": v} for k, v in sorted(skipped_counts.items())
        ])
        qc_path = results_path / "analysis" / "plot_generation_skip_qc.csv"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_df.to_csv(qc_path, index=False)
        print(f"[QC] Skip summary: {qc_path}")
    if run_csv_audit:
        audit_csv_quality(results_path)
    if run_png_audit:
        audit_png_quality(results_path)
    if run_proposal_artifact_audit:
        audit_proposal_required_artifacts(results_path)
    if run_proposal_metric_audit:
        audit_proposal_metric_expectations(results_path)
    if write_proposal_map:
        write_proposal_expectation_mapping(results_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate high-quality plots from experiment CSVs')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory containing CSV files')
    parser.add_argument('--include-static-only', action='store_true',
                        help='Also build generic training plots for experiments that already have canonical static plots')
    parser.add_argument('--overwrite-duplicates', action='store_true',
                        help='Overwrite duplicate 2D/SAM/robustness PNG names with canonical versions before plotting')
    parser.add_argument('--sync-peer-dirs', type=str, default='',
                        help='Comma-separated peer result roots to sync from --results-dir canonical PNGs')
    parser.add_argument('--no-cleanup-stale', action='store_true',
                        help='Do not remove stale generic training-results PNG/CSV files for static-only experiments')
    parser.add_argument('--no-csv-audit', action='store_true',
                        help='Do not generate CSV quality audit reports')
    parser.add_argument('--no-png-audit', action='store_true',
                        help='Do not generate PNG quality audit reports')
    parser.add_argument('--no-proposal-audit', action='store_true',
                        help='Do not generate proposal-required artifact audit reports')
    parser.add_argument('--no-proposal-metric-audit', action='store_true',
                        help='Do not generate proposal-required metric/schema audit reports')
    parser.add_argument('--no-proposal-map', action='store_true',
                        help='Do not write proposal expectation mapping reports')

    args = parser.parse_args()

    sync_peer_dirs = [p.strip() for p in args.sync_peer_dirs.split(',') if p.strip()]

    generate_all_plots(
        args.results_dir,
        include_static_only=args.include_static_only,
        overwrite_duplicates=args.overwrite_duplicates,
        sync_peer_dirs=sync_peer_dirs,
        cleanup_stale=(not args.no_cleanup_stale),
        run_csv_audit=(not args.no_csv_audit),
        run_png_audit=(not args.no_png_audit),
        run_proposal_artifact_audit=(not args.no_proposal_audit),
        run_proposal_metric_audit=(not args.no_proposal_metric_audit),
        write_proposal_map=(not args.no_proposal_map),
    )
