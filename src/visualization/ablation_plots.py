"""
Shared Visualization Utilities for Ablation Studies

Provides reusable plotting functions for:
- Advanced training ablation
- Initialization ablation  
- Comprehensive ablation studies

All functions generate high-quality plots in PNG and PDF formats.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def create_ablation_bar_plot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    baseline_name: str = 'Baseline',
    figsize: Tuple[int, int] = (12, 6),
    color_scheme: Optional[Dict[str, str]] = None
):
    """
    Create a bar plot comparing different ablation configurations.
    
    Args:
        df: DataFrame with results
        group_col: Column name for grouping (e.g., 'configuration')
        value_col: Column name for values (e.g., 'test_accuracy')
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save figure (without extension)
        baseline_name: Name of baseline configuration for coloring
        figsize: Figure size (width, height)
        color_scheme: Optional dict mapping config names to colors
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group and sort
    grouped = df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
    # Use explicit keyword args for clarity and typing
    grouped = grouped.sort_values(by='mean', ascending=False)
    
    x_pos = np.arange(len(grouped))
    bars = ax.bar(x_pos, np.asarray(grouped['mean'], dtype=float), yerr=np.asarray(grouped['std'], dtype=float),
                  capsize=5, alpha=0.7, edgecolor='black')
    
    # Color coding
    if color_scheme is None:
        colors = []
        for config in grouped.index:
            if config == baseline_name:
                colors.append('#95a5a6')  # Gray for baseline
            elif '+' in str(config):
                colors.append('#2ecc71')  # Green for combinations
            else:
                colors.append('#3498db')  # Blue for single features
    else:
        colors = [color_scheme.get(config, '#3498db') for config in grouped.index]
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in grouped.index], rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean_val, std_val) in enumerate(zip(grouped['mean'], grouped['std'])):
        ax.text(i, mean_val + std_val + max(grouped['mean'])*0.01, 
               f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    
    return fig


def create_box_plot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 6)
):
    """Create box plot showing distribution across seeds."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    from src.utils.plot_helpers import arr_to_numpy_float
    groups = [str(g) for g in df[group_col].unique()]
    box_data = [arr_to_numpy_float(df[df[group_col] == group][value_col]) for group in groups]
    
    # Use labels=groups and ensure numeric arrays for plotting
    bp = ax.boxplot(box_data, patch_artist=True,
                   showmeans=True, meanline=True)
    
    # Style boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    
    # Ensure tick labels are strings and set them explicitly
    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha='right')
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    
    return fig


def create_improvement_heatmap(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    baseline_name: str,
    features: List[str],
    title: str,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create heatmap showing which features are active and their improvements.
    
    Args:
        features: List of feature names to check for in configuration names
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate baseline
    baseline_value = df[df[group_col] == baseline_name][value_col].mean()
    
    # Get non-baseline configurations
    configs = [c for c in df[group_col].unique() if c != baseline_name]
    
    # Build feature matrix
    feature_matrix = []
    improvements = []
    
    for config in configs:
        row = [int(feat in str(config)) for feat in features]
        feature_matrix.append(row)
        
        config_value = df[df[group_col] == config][value_col].mean()
        improvements.append(config_value - baseline_value)
    
    if not feature_matrix:
        plt.close()
        return None
    
    feature_matrix = np.array(feature_matrix)
    
    # Create heatmap
    im = ax.imshow(feature_matrix.T, cmap='RdYlGn', aspect='auto',
                  vmin=0, vmax=1, alpha=0.6)
    
    # Set ticks
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    
    # Add improvement values
    for i, improvement in enumerate(improvements):
        color = 'green' if improvement > 0 else 'red'
        ax.text(i, -0.5, f'{improvement:+.2f}%',
               ha='center', va='top', fontsize=9, fontweight='bold',
               color=color)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Active Features', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Grid
    ax.set_xticks(np.arange(len(configs)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(features)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    
    return fig


def create_summary_table(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    baseline_name: str,
    title: str,
    output_path: Path,
    additional_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8)
):
    """Create visual summary table."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Compute baseline
    baseline_value = df[df[group_col] == baseline_name][value_col].mean()
    
    # Prepare data
    grouped = df.groupby(group_col).agg({
        value_col: ['mean', 'std', 'count']
    })
    
    summary_data = []
    for config in grouped.index:
        mean = grouped.loc[config, (value_col, 'mean')]
        std = grouped.loc[config, (value_col, 'std')]
        count = grouped.loc[config, (value_col, 'count')]
        improvement = mean - baseline_value
        
        row = [
            config,
            f"{mean:.2f} ± {std:.2f}",
            f"{improvement:+.2f}",
            f"{int(count)} seeds"
        ]
        
        summary_data.append(row)
    
    table = ax.table(
        cellText=summary_data,
        colLabels=['Configuration', 'Mean ± Std', 'vs Baseline', 'Samples'],
        cellLoc='left',
        loc='center',
        colWidths=[0.35, 0.25, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color rows
    for i, config in enumerate(grouped.index):
        color = 'lightgray' if config == baseline_name else 'lightblue'
        for j in range(4):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    # Style header
    for j in range(4):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    
    return fig


def generate_all_ablation_plots(
    df: pd.DataFrame,
    results_dir: str,
    study_name: str,
    group_col: str = 'configuration',
    value_col: str = 'test_accuracy',
    baseline_name: str = 'Baseline',
    features: Optional[List[str]] = None
) -> Path:
    """
    Generate complete set of ablation study visualizations.
    
    Args:
        df: Results DataFrame
        results_dir: Base directory for results
        study_name: Name of study (e.g., 'advanced_training', 'initialization')
        group_col: Column name for grouping
        value_col: Column name for metric
        baseline_name: Name of baseline configuration
        features: List of feature names for heatmap (optional)
    
    Returns:
        Path to visualization directory
    """
    viz_dir = Path(results_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations for {study_name} study...")
    
    # 1. Bar plot
    create_ablation_bar_plot(
        df=df,
        group_col=group_col,
        value_col=value_col,
        title=f'{study_name.replace("_", " ").title()}: Ablation Study Results',
        ylabel=value_col.replace('_', ' ').title() + ' (%)',
        output_path=viz_dir / 'accuracy_comparison',
        baseline_name=baseline_name
    )
    print("  accuracy_comparison.png/.pdf")
    
    # 2. Box plot
    create_box_plot(
        df=df,
        group_col=group_col,
        value_col=value_col,
        title=f'{study_name.replace("_", " ").title()}: Distribution Across Seeds',
        ylabel=value_col.replace('_', ' ').title() + ' (%)',
        output_path=viz_dir / 'accuracy_distribution'
    )
    print("  accuracy_distribution.png/.pdf")
    
    # 3. Heatmap (if features provided)
    if features:
        create_improvement_heatmap(
            df=df,
            group_col=group_col,
            value_col=value_col,
            baseline_name=baseline_name,
            features=features,
            title=f'Feature Activation & Performance Impact',
            output_path=viz_dir / 'feature_heatmap'
        )
        print("  feature_heatmap.png/.pdf")
    
    # 4. Summary table
    create_summary_table(
        df=df,
        group_col=group_col,
        value_col=value_col,
        baseline_name=baseline_name,
        title=f'{study_name.replace("_", " ").title()}: Summary Table',
        output_path=viz_dir / 'summary_table'
    )
    print("  summary_table.png/.pdf")
    
    print(f"All visualizations saved to {viz_dir}/\n")
    
    return viz_dir
