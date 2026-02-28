#!/usr/bin/env python3
"""
Universal visualization generator for all experiments.
Reads CSV files from results/ and automatically generates high-quality plots.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import seaborn as sns
import traceback
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10



def _get_x_axis(df):
    """Robustly detect x-axis column (prioritizing epoch > iteration > index)."""
    # Look for 'epoch' or anything containing it
    epoch_col = next((col for col in df.columns if col.strip().lower() == 'epoch' or 'epoch' in col.lower()), None)
    if epoch_col is not None:
        return df[epoch_col].values, "Epoch"
    
    # Fallback to iteration
    iter_col = next((col for col in df.columns if 'iter' in col.lower()), None)
    if iter_col is not None:
        return df[iter_col].values, "Iteration"
        
    # Final fallback: row index
    return np.arange(1, len(df) + 1), "Epoch"


def plot_training_curves(csv_files: List[str], output_dir: Path, title: str = "Training Curves"):
    """
    Generate training curves from CSV files.
    Handles MNIST, CIFAR-10, NLP, and other NN experiments.
    """
    if not csv_files:
        return

    # Patterns that indicate meta/aggregator CSV files (not actual per-epoch training runs).
    # Any file starting with 'advablation_' is a combined ablation summary - skip it.
    _META_PATTERNS = [
        'csv_qc', 'ablation_summary', 'convergence_rate_summary',
        'lr_schedule_demo', 'advanced_ablation_results',
        'advablation', # catches advablation_baseline, advablation_all, etc.
    ]

    # Group by optimizer
    results = {}
    for csv_file in csv_files:
        basename = os.path.basename(csv_file).lower()

        # Skip meta/aggregator files - they are not training runs
        # Refined: only skip if it matches a meta pattern AND does not look like a training run (no _seed/_start/_trial)
        if any(pat in basename for pat in _META_PATTERNS) and not any(run_pat in basename for run_pat in ['_seed', '_start', '_trial']):
            if 'other' in title.lower():
                print(f"[DEBUG] Skipping meta file: {basename}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_file)

        # Check file has actual training data (must have loss or accuracy columns)
        has_training_data = any(
            col for col in df.columns
            if any(kw in col.lower() for kw in ['loss', 'acc', 'accuracy'])
        )
        if not has_training_data:
            continue

        # Extract label from filename (removes trial/seed suffixes)
        base = os.path.basename(csv_file).replace('.csv', '')
        import re
        optimizer = re.sub(r'(_start|_trial|_seed|_run)\d+$', '', base, flags=re.IGNORECASE)
        
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
            if 'other' in title.lower():
                print(f"[DEBUG] Skipping identified 'Unknown' for: {basename}", file=sys.stderr)
            continue

        if optimizer not in results:
            results[optimizer] = []
        results[optimizer].append(df)

    # Normalize and sort optimizers for stable color assignment
    sorted_optimizers = sorted(results.keys())
    
    # Define stable colors
    base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FFD93D', '#B8E6F1', 
                   '#A29BFE', '#FAB1A0', '#55E6C1', '#25CCF7', '#FD7272', '#58B19F', '#BDC581']
    opt_to_color = {opt: base_colors[i % len(base_colors)] for i, opt in enumerate(sorted_optimizers)}

    # DEBUG: trace what landed in results
    print(f'[DEBUG] plot_training_curves({title!r}): optimizers={sorted_optimizers!r}', file=sys.stderr)

    # Create figure - significantly larger to prevent cramming
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

    # Plot 1: Training Loss
    ax = axes[0, 0]
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
                ax.plot(run_df['x'], run_df['y'], color=color, alpha=0.3, linewidth=1)
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
            if max_x < min_x or max_x - min_x > 100000: # Prevent massive arange calls
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
                except:
                    continue
            
            if aligned_losses:
                mean_loss = np.mean(np.vstack(aligned_losses), axis=0)
                ax.plot(common_x, mean_loss, color=color, linewidth=2.5, label=optimizer)

    ax.set_xlabel(x_label_for_plot, fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss Curves', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    # Move legend outside the plot area
    ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Plot 2: Test/Validation Accuracy
    ax = axes[0, 1]
    has_test_acc = False
    for i, (optimizer, dfs) in enumerate(sorted(results.items())):
        color = opt_to_color[optimizer]
        
        runs_for_mean = []
        x_label_for_plot = "Epoch" # Default

        for df in dfs:
            x_vals, x_label = _get_x_axis(df)
            x_label_for_plot = x_label
            
            # Detect accuracy column
            acc_col = next((col for col in df.columns if 'test' in col.lower() and ('acc' in col.lower() or 'accuracy' in col.lower())), None)
            
            if acc_col:
                acc_vals = pd.to_numeric(df[acc_col], errors='coerce')
                # Filter out dummy/placeholder values
                if acc_vals.isna().all() or (acc_vals == 0.01).all() or (acc_vals == 0).all() or (acc_vals == 0.5).all():
                    continue
                    
                # Convert to percentage if needed
                if acc_vals.max() <= 1.01:
                    acc_vals = acc_vals * 100.0
                
                has_test_acc = True
                run_df = pd.DataFrame({'x': x_vals, 'y': acc_vals}).groupby('x').mean().reset_index()
                ax.plot(run_df['x'], run_df['y'], color=color, alpha=0.3, linewidth=1)
                runs_for_mean.append((run_df['x'].values, run_df['y'].values))

        # Mean line
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
                except:
                    continue
            
            if aligned_accs:
                mean_acc = np.mean(np.vstack(aligned_accs), axis=0)
                ax.plot(common_x, mean_acc, color=color, linewidth=2.5, label=optimizer)

    ax.set_xlabel(x_label_for_plot, fontsize=14, fontweight='bold')
    if has_test_acc:
        ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Test Accuracy', fontsize=16, fontweight='bold', pad=15)
        # Move legend outside the plot area
        ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    else:
        ax.set_ylabel('N/A', fontsize=14, fontweight='bold')
        ax.set_title('Test Accuracy (N/A for Task)', fontsize=16, fontweight='bold', pad=15)
        ax.text(0.5, 0.5, 'N/A: Optimization Task', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # Plot 3: Final Performance Bar Chart
    ax = axes[1, 0]
    final_metrics = {}
    final_stds = {}

    for optimizer, dfs in results.items():
        # Get final test accuracy from each run
        final_vals = []
        for df in dfs:
            acc_col = next((col for col in df.columns if 'acc' in col.lower() and 'test' in col.lower()), None)
            if acc_col:
                val = df[acc_col].iloc[-1]
                if val <= 1.0:
                    val = val * 100
                final_vals.append(val)

        if final_vals:
            final_metrics[optimizer] = np.mean(final_vals)
            final_stds[optimizer] = np.std(final_vals) if len(final_vals) > 1 else 0

    # Filter final_metrics to REMOVE any persistent Unknown entries
    final_metrics = {k: v for k, v in final_metrics.items() if str(k).lower() not in ('unknown', 'nan', '', 'missing')}

    if final_metrics:
        # Sort for better presentation
        optimizers_sorted = sorted(final_metrics.keys(), key=lambda k: final_metrics[k], reverse=True)
        x_pos = np.arange(len(optimizers_sorted))
        bars = ax.bar(x_pos, [final_metrics[opt] for opt in optimizers_sorted],
                      yerr=[final_stds[opt] for opt in optimizers_sorted],
                      color=[opt_to_color[opt] for opt in optimizers_sorted],
                      alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizers_sorted, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Final Test Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Final Performance', fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for bar, opt in zip(bars, optimizers_sorted):
            height = bar.get_height()
            # If the bar represents 0 standard deviation over identical runs, just plot it
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{final_metrics[opt]:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 4: Training speed comparison
    ax = axes[1, 1]
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
                    
            if elapsed is not None:
                # Detect dummy timing data: if elapsed seconds exactly matches row count
                # Or if time is uniform across exactly equal step sizes, making it artificial
                if abs(elapsed - len(df)) < 1e-3:
                    continue
                    
                # Another fail-safe: if max time is literally 1.0 or 10.0 and len is 100/1000 etc
                if elapsed % 1 == 0 and len(df) % 10 == 0 and df['elapsed_seconds'].nunique() <= 2:
                    # Possibly uniform synthetic array, skip
                    continue
                    
                # Throughput: logical steps or epochs per cumulative time.
                # Using max_x prevents huge discrepancies between batch-level and epoch-level logging.
                x_vals, _ = _get_x_axis(df)
                if len(x_vals) > 0 and elapsed > 1e-4:
                    max_x = np.max(x_vals)
                    speed = max_x / elapsed
                    run_speeds.append(speed)
                    
        # Only compute mean if we have valid non-dummy runs
        if run_speeds:
            speeds[optimizer] = np.mean(run_speeds)

    if speeds and len(speeds) > 0:
        # Sort keys based on speed values descending
        sorted_keys = sorted(speeds.keys(), key=lambda k: speeds[k], reverse=True)
        # Check if they are all identically equal (dummy failure case)
        vals = [speeds[k] for k in sorted_keys]
        if max(vals) - min(vals) < 1e-5 and max(vals) == 1000.0:
            # Everything is identical dummy speed, ignore it
            ax.text(0.5, 0.5, 'All timing data was synthetic placeholder.\nEfficiency N/A',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
        else:
            x_pos = np.arange(len(sorted_keys))
            bars = ax.bar(x_pos, [speeds[k] for k in sorted_keys],
                          color=[opt_to_color[k] for k in sorted_keys],
                          alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sorted_keys, rotation=45, ha='right', fontsize=11)
            ax.set_ylabel('Speed (Epochs or Steps / Sec)', fontsize=14, fontweight='bold')
            ax.set_title('Training Efficiency', fontsize=16, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid timing data available',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)

    plt.tight_layout()
    output_file = output_dir / f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

    # Generate Tabular Summary
    try:
        summary_data = []
        for optimizer, dfs in results.items():
            acc = final_metrics.get(optimizer, np.nan)
            acc_std = final_stds.get(optimizer, np.nan)
            speed = speeds.get(optimizer, np.nan)
            
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


def generate_all_plots(results_dir: str = 'results'):
    """
    Automatically generate plots for all experiments in results directory.
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    print(f"[PLOTS] Generating visualizations from: {results_dir}")
    print("="*80)

    # Find all CSV files
    try:
        all_csvs = glob.glob(str(results_path / "**/*.csv"), recursive=True)
    except Exception as e:
        print(f"[WARN] Error finding CSV files: {e}")
        all_csvs = []

    if not all_csvs:
        print("[WARN] No CSV files found")
        return

    # Find all baseline files first to share them across relevant groups
    baseline_csvs = [f for f in all_csvs if 'baseline' in os.path.basename(f).lower()]
    print(f"Found {len(baseline_csvs)} baseline CSV files for cross-referencing")

    # Group by experiment type dynamically
    experiments = {}

    for csv_file in all_csvs:
        path_obj = Path(csv_file)
        path_str = str(path_obj).upper()
        
        category = None
        # Specific check for core known tasks
        if '2D' in path_str or 'OPTIMIZATION' in path_str:
            category = '2D Optimization'
        elif 'MNIST' in path_str:
            category = 'MNIST'
        elif 'CIFAR10' in path_str:
            category = 'CIFAR10'
        elif 'IMDB' in path_str:
            category = 'IMDB'
        elif 'MEDICAL' in path_str:
            category = 'Medical'
        else:
            # Parse the folder name directly from 'experiments/XYZ/...'
            parts = path_obj.parts
            # We want to find the first folder after 'experiments' or similar root
            try:
                exp_idx = next(i for i, p in enumerate(parts) if p.lower() == 'experiments')
                if exp_idx + 1 < len(parts) and not str(parts[exp_idx + 1]).endswith('.csv'):
                    sub_dir = parts[exp_idx + 1]
                    category = sub_dir.replace('_', ' ').title()
            except StopIteration:
                # If 'experiments' isn't in path, but 'results' is
                try:
                    res_idx = next(i for i, p in enumerate(parts) if p.lower() == 'results_proposal_full_20260223_v2')
                    if res_idx + 1 < len(parts) and not str(parts[res_idx + 1]).endswith('.csv'):
                        sub_dir = parts[res_idx + 1]
                        category = sub_dir.replace('_', ' ').title()
                except StopIteration:
                    pass

            if not category:
                category = 'Other'
                
        if category not in experiments:
            experiments[category] = []
            
        experiments[category].append(csv_file)

    # Inject baseline into relevant ablation categories (primarily MNIST-based ones)
    for exp_type in experiments.keys():
        is_mnist_ablation = any(kw in exp_type.lower() for kw in ['ablation', 'sensitivity', 'mnist'])
        is_baseline_group = 'baseline' in exp_type.lower() or 'advanced ablation' in exp_type.lower()
        
        # Only inject if it's an ablation group and NOT the group the baseline originally belongs to
        if is_mnist_ablation and not is_baseline_group:
            existing_basenames = [os.path.basename(f) for f in experiments[exp_type]]
            for b_file in baseline_csvs:
                if os.path.basename(b_file) not in existing_basenames:
                    experiments[exp_type].append(b_file)

    # Generate plots for each category
    viz_dir = results_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    plots_created = 0
    for exp_type, csv_files in experiments.items():
        if csv_files:
            print(f"\n[GROUP] {exp_type}: {len(csv_files)} files")
            # We already grouped by exp_type, so plot_training_curves will handle them consistently.
            # To ensure different sub-experiments inside "Other" don't mix, 
            # we can optionally pass a prefix or just trust the better categorization.
            try:
                plot_training_curves(csv_files, viz_dir, title=f"{exp_type} Training Results")
                plots_created += 1
            except Exception as e:
                import traceback
                print(f"   [WARN] Error in {exp_type}: {e}")
                traceback.print_exc()

    print("\n" + "="*80)
    print(f"[OK] Created {plots_created} visualization sets in: {viz_dir}")
    print(f"   All plots are high-quality (300 DPI)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate high-quality plots from experiment CSVs')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory containing CSV files')

    args = parser.parse_args()

    generate_all_plots(args.results_dir)
