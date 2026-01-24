#!/usr/bin/env python3
"""Checkpoint & Visualization sanity checker

- Scans a results directory for experiment CSVs, existing visualizations, and checkpoints (.pt/.pth)
- For each discovered experiment it verifies that the expected static and interactive visualization
  artifacts exist and, if requested or missing, regenerates them from available CSVs.

Usage:
    python scripts/checkpoint_and_viz_check.py --results-dir results [--force] [--experiments MNIST,CIFAR10]

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List


def infer_experiment_name_from_stem(stem: str) -> str:
    s = stem.upper()
    if 'MNIST' in s:
        return 'MNIST'
    if 'CIFAR' in s:
        return 'CIFAR10'
    if s.startswith('NN_'):
        parts = stem.split('_')
        # pattern: NN_<Model>_<Dataset>_<Optimizer>_...
        if len(parts) >= 3:
            return parts[2]
        if len(parts) >= 2:
            return parts[1]
    # model-like names
    if 'RESNET' in s:
        return 'ResNet18'
    # fallback to first part of stem
    return stem.split('_')[0].capitalize()


def detect_csv_experiments(results_dir: Path) -> Dict[str, List[Path]]:
    csvs = list(results_dir.rglob('*.csv'))
    experiments: Dict[str, List[Path]] = {}
    for p in csvs:
        stem = p.stem
        name = infer_experiment_name_from_stem(stem)
        experiments.setdefault(name, []).append(p)
    return experiments


def detect_visualization_status(results_dir: Path, experiment_name: str) -> Dict[str, bool]:
    viz_dir = results_dir / 'visualizations'
    static_dir = viz_dir / 'static' / experiment_name.lower()
    interactive_dir = viz_dir / 'interactive'

    expected_static = [
        static_dir / f"{experiment_name.lower()}_train_loss.png",
        static_dir / f"{experiment_name.lower()}_test_accuracy.png",
        static_dir / f"{experiment_name.lower()}_final_comparison.png",
    ]
    expected_interactive = [interactive_dir / f"{experiment_name.lower()}_interactive_comparison.html"]

    status = {}
    for p in expected_static + expected_interactive:
        status[str(p)] = p.exists()
    return status


def detect_checkpoints(results_dir: Path) -> List[Path]:
    patterns = ['*.pt', '*.pth']
    found = []
    for pat in patterns:
        found.extend(results_dir.rglob(pat))
    return found


def ensure_visualizations(results_dir: Path, experiments: Dict[str, List[Path]], *, force: bool = False) -> int:
    """For each experiment, check visuals and regenerate missing ones when CSVs exist.

    Returns non-zero on partial failure.
    """
    try:
        # Import here to avoid heavy imports unless this script runs
        from runners.run_all_kaggle import create_experiment_visualizations
    except Exception as e:
        logging.error("Could not import visualization helper: %s", e)
        return 2

    exit_code = 0
    checkpoints = detect_checkpoints(results_dir)

    for exp_name, csv_list in experiments.items():
        status = detect_visualization_status(results_dir, exp_name)
        missing = [p for p, ok in status.items() if not ok]

        if not missing and not force:
            print(f"✅ Visualizations for '{exp_name}' appear complete")
            continue

        print(f"\nChecking '{exp_name}': {len(csv_list)} CSV(s) found, {len(missing)} missing visual(s)")

        if csv_list:
            # regenerate visuals using CSVs
            try:
                create_experiment_visualizations(exp_name, str(results_dir), csv_list)
                # Re-check
                status2 = detect_visualization_status(results_dir, exp_name)
                still_missing = [p for p, ok in status2.items() if not ok]
                if still_missing:
                    print(f"⚠️ After generation, still missing: {still_missing}")
                    exit_code = 1
                else:
                    print(f"✅ Regenerated missing visuals for '{exp_name}'")
            except Exception as e:
                logging.exception("Failed to generate visuals for %s: %s", exp_name, e)
                exit_code = 1
        else:
            if checkpoints:
                print(f"⚠️ No CSVs for '{exp_name}', but checkpoints were found under {results_dir}.")
                print("   The current auto-generation path requires CSVs. You can either:")
                print("    - generate CSVs from checkpoints using your experiment runner or analysis scripts")
                print("    - or run the visualization scripts that support loading checkpoints directly (see src/visualization)")
                exit_code = 1
            else:
                print(f"⚠️ No CSVs or checkpoints found for '{exp_name}'; cannot generate visualizations")
                exit_code = 1

    return exit_code


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check and regenerate missing visualizations from CSVs")
    parser.add_argument('--results-dir', '-r', default='results', help='Base results directory (default: results)')
    parser.add_argument('--experiments', '-e', help='Comma-separated list of experiments to check (default: all inferred from CSVs)')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if visualization files exist')
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logging.error("Results directory does not exist: %s", results_dir)
        return 2

    experiments: Dict[str, List[Path]] = detect_csv_experiments(results_dir)

    if args.experiments:
        wanted = [s.strip() for s in args.experiments.split(',') if s.strip()]
        experiments = {k: v for k, v in experiments.items() if k in wanted}
        # If user requested experiments but none of the CSVs matched, still include empty entries
        for w in wanted:
            experiments.setdefault(w, [])

    if not experiments:
        print("No CSVs found in the results directory. Nothing to do.")
        return 0

    print(f"Found experiments: {', '.join(sorted(experiments.keys()))}")

    rc = ensure_visualizations(results_dir, experiments, force=args.force)
    if rc == 0:
        print('\nAll checks complete ✅')
    else:
        print('\nCompleted with warnings. See messages above.')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
