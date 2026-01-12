#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Results Aggregation and Report Generation
===================================================

Scans result directories, aggregates CSVs, generates comprehensive reports.

Usage:
    python scripts/aggregate_results.py --results-dir results/ --output report.md
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def find_result_csvs(results_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all result CSV files organized by experiment.

    Returns:
        Dict mapping experiment name to list of CSV paths
    """
    experiment_csvs = {}

    for csv_path in results_dir.rglob("*.csv"):
        # Skip temporary/backup files
        if any(x in csv_path.name for x in ['backup', 'temp', 'PARTIAL']):
            continue

        # Determine experiment type from path or filename
        experiment = None
        if 'mnist' in str(csv_path).lower():
            experiment = 'MNIST'
        elif 'cifar' in str(csv_path).lower():
            experiment = 'CIFAR10'
        elif 'nlp' in str(csv_path).lower() or 'imdb' in str(csv_path).lower():
            experiment = 'NLP'
        elif 'medical' in str(csv_path).lower():
            experiment = 'Medical'
        elif 'resnet' in str(csv_path).lower():
            experiment = 'ResNet'
        elif '2d' in str(csv_path).lower():
            experiment = '2D'
        else:
            experiment = 'Other'

        if experiment not in experiment_csvs:
            experiment_csvs[experiment] = []
        experiment_csvs[experiment].append(csv_path)

    return experiment_csvs


def aggregate_experiment_results(csv_paths: List[Path]) -> pd.DataFrame:
    """
    Aggregate multiple CSV files into a single DataFrame.

    Args:
        csv_paths: List of CSV file paths

    Returns:
        Aggregated DataFrame
    """
    dfs = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Could not load {csv_path}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def compute_statistics(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    """
    Compute statistics for a metric across seeds.

    Args:
        df: DataFrame with optimizer and metric columns
        metric: Column name for metric

    Returns:
        Dict with mean, std, min, max, median
    """
    if metric not in df.columns:
        return {}

    stats = {
        'mean': float(df[metric].mean()),
        'std': float(df[metric].std()),
        'min': float(df[metric].min()),
        'max': float(df[metric].max()),
        'median': float(df[metric].median())
    }

    return stats


def generate_optimizer_comparison_table(df: pd.DataFrame, metric: str) -> str:
    """
    Generate markdown table comparing optimizers.

    Args:
        df: Aggregated results DataFrame
        metric: Metric to compare (e.g., 'final_test_acc')

    Returns:
        Markdown table string
    """
    if 'optimizer' not in df.columns or metric not in df.columns:
        return "No data available for comparison"

    # Group by optimizer and compute statistics
    grouped = df.groupby('optimizer')[metric].agg(['mean', 'std', 'min', 'max', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)

    # Generate markdown table
    table = f"| Optimizer | Mean {metric} | Std | Min | Max | Seeds |\n"
    table += "|-----------|---------------|-----|-----|-----|-------|\n"

    for optimizer, row in grouped.iterrows():
        table += f"| {optimizer} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} | {int(row['count'])} |\n"

    return table


def generate_convergence_summary(df: pd.DataFrame) -> str:
    """
    Generate convergence analysis summary.

    Args:
        df: Results DataFrame

    Returns:
        Markdown formatted summary
    """
    summary = "## Convergence Summary\n\n"

    if 'training_time' in df.columns:
        summary += "### Training Time\n\n"
        summary += generate_optimizer_comparison_table(df, 'training_time')
        summary += "\n\n"

    if 'epochs_completed' in df.columns:
        summary += "### Epochs Completed\n\n"
        summary += generate_optimizer_comparison_table(df, 'epochs_completed')
        summary += "\n\n"

    return summary


def generate_robust_gradient_summary(df: pd.DataFrame) -> str:
    """
    Generate robust gradient handling summary.

    Args:
        df: Results DataFrame

    Returns:
        Markdown formatted summary
    """
    summary = "## Robust Gradient Statistics\n\n"

    gradient_metrics = ['mean_grad_norm', 'max_grad_norm', 'clip_fraction', 'heavy_tail_fraction']
    available_metrics = [m for m in gradient_metrics if m in df.columns]

    if not available_metrics:
        summary += "*No robust gradient statistics available (not enabled in this run)*\n\n"
        return summary

    summary += "### Gradient Norms and Clipping\n\n"

    for metric in available_metrics:
        summary += f"#### {metric.replace('_', ' ').title()}\n\n"
        summary += generate_optimizer_comparison_table(df, metric)
        summary += "\n\n"

    return summary


def generate_markdown_report(
    results_dir: Path,
    output_path: Path,
    experiment_data: Dict[str, pd.DataFrame]
) -> None:
    """
    Generate comprehensive markdown report.

    Args:
        results_dir: Results directory path
        output_path: Output markdown file path
        experiment_data: Dict mapping experiment name to DataFrame
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# GDSearch Comprehensive Results Report\n\n")
        f.write(f"**Generated from:** `{results_dir}`\n\n")
        f.write("---\n\n")

        # Table of Contents
        f.write("## Table of Contents\n\n")
        for experiment in sorted(experiment_data.keys()):
            f.write(f"- [{experiment}](#{experiment.lower().replace(' ', '-')})\n")
        f.write("\n---\n\n")

        # Experiment sections
        for experiment, df in sorted(experiment_data.items()):
            f.write(f"## {experiment}\n\n")

            if df.empty:
                f.write("*No data available*\n\n")
                continue

            f.write(f"**Total runs:** {len(df)}\n\n")

            if 'optimizer' in df.columns:
                optimizers = df['optimizer'].unique()
                f.write(f"**Optimizers:** {', '.join(optimizers)}\n\n")

            if 'seed' in df.columns:
                seeds = df['seed'].unique()
                f.write(f"**Seeds:** {len(seeds)}\n\n")

            # Test accuracy comparison
            if 'final_test_acc' in df.columns:
                f.write("### Test Accuracy\n\n")
                f.write(generate_optimizer_comparison_table(df, 'final_test_acc'))
                f.write("\n\n")

            # Test loss comparison
            if 'final_test_loss' in df.columns:
                f.write("### Test Loss\n\n")
                f.write(generate_optimizer_comparison_table(df, 'final_test_loss'))
                f.write("\n\n")

            # Medical-specific: Dice coefficient
            if 'final_test_dice' in df.columns:
                f.write("### Dice Coefficient\n\n")
                f.write(generate_optimizer_comparison_table(df, 'final_test_dice'))
                f.write("\n\n")

            # Convergence summary
            f.write(generate_convergence_summary(df))

            # Robust gradient summary
            f.write(generate_robust_gradient_summary(df))

            f.write("---\n\n")

    logging.info(f"Report saved to: {output_path}")


def generate_json_summary(
    results_dir: Path,
    output_path: Path,
    experiment_data: Dict[str, pd.DataFrame]
) -> None:
    """
    Generate JSON summary for programmatic access.

    Args:
        results_dir: Results directory path
        output_path: Output JSON file path
        experiment_data: Dict mapping experiment name to DataFrame
    """
    summary = {
        'results_dir': str(results_dir),
        'experiments': {}
    }

    for experiment, df in experiment_data.items():
        if df.empty:
            summary['experiments'][experiment] = {'status': 'no_data'}
            continue

        exp_summary = {
            'total_runs': len(df),
            'optimizers': list(df['optimizer'].unique()) if 'optimizer' in df.columns else [],
            'seeds': list(df['seed'].unique()) if 'seed' in df.columns else [],
            'metrics': {}
        }

        # Aggregate key metrics
        metrics = ['final_test_acc', 'final_test_loss', 'final_test_dice', 'training_time', 'epochs_completed']
        for metric in metrics:
            if metric in df.columns:
                exp_summary['metrics'][metric] = compute_statistics(df, metric)

        # Robust gradient statistics
        gradient_metrics = ['mean_grad_norm', 'max_grad_norm', 'clip_fraction', 'heavy_tail_fraction']
        exp_summary['robust_gradients'] = {}
        for metric in gradient_metrics:
            if metric in df.columns:
                exp_summary['robust_gradients'][metric] = compute_statistics(df, metric)

        summary['experiments'][experiment] = exp_summary

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"JSON summary saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Aggregate GDSearch experiment results and generate reports"
    )

    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory to scan (default: results/)')
    parser.add_argument('--output', type=str, default='results_report.md',
                        help='Output markdown report file (default: results_report.md)')
    parser.add_argument('--json-output', type=str, default='results_summary.json',
                        help='Output JSON summary file (default: results_summary.json)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logging.error(f"Results directory not found: {results_dir}")
        return 1

    logging.info(f"Scanning results directory: {results_dir}")

    # Find all result CSVs
    experiment_csvs = find_result_csvs(results_dir)

    if not experiment_csvs:
        logging.warning("No result CSV files found")
        return 1

    logging.info(f"Found {len(experiment_csvs)} experiments:")
    for experiment, csvs in experiment_csvs.items():
        logging.info(f"  {experiment}: {len(csvs)} CSV files")

    # Aggregate results for each experiment
    experiment_data = {}
    for experiment, csv_paths in experiment_csvs.items():
        df = aggregate_experiment_results(csv_paths)
        experiment_data[experiment] = df
        logging.info(f"Aggregated {experiment}: {len(df)} rows")

    # Generate markdown report
    output_path = Path(args.output)
    generate_markdown_report(results_dir, output_path, experiment_data)

    # Generate JSON summary
    json_output_path = Path(args.json_output)
    generate_json_summary(results_dir, json_output_path, experiment_data)

    logging.info("Aggregation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
