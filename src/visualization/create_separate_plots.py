"""
Script to create separate high-resolution plots from experiment results.
Each plot is saved as an individual PNG file for easy viewing and presentation.

Usage:
    python src/visualization/create_separate_plots.py
    
Output:
    6 separate plots in plots/ directory:
    - 01_final_loss_comparison.png
    - 02_distance_to_optimum.png
    - 03_convergence_rate.png
    - 04_loss_distribution_boxplot.png
    - 05_statistical_significance_heatmap.png
    - 06_effect_sizes.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def create_separate_plots(
    summary_csv='results/optimizer_summary.csv',
    stats_csv='results/statistical_comparisons.csv',
    detailed_csv='results/multiseed_detailed.csv',
    output_dir='plots'
):
    """
    Create 6 separate high-quality visualizations from experiment results.
    
    Parameters:
    -----------
    summary_csv : str
        Path to optimizer summary CSV file
    stats_csv : str
        Path to statistical comparisons CSV file
    detailed_csv : str
        Path to detailed multi-seed results CSV file
    output_dir : str
        Directory to save output plots
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    
    # Read data
    print("📊 Reading experiment data...")
    summary_df = pd.read_csv(summary_csv)
    stats_df = pd.read_csv(stats_csv)
    detailed_df = pd.read_csv(detailed_csv)
    
    # Extract data
    optimizers = summary_df['Optimizer'].values
    losses = summary_df['Mean Loss'].values
    stds = summary_df['Std Loss'].values
    distances = summary_df['Mean Distance'].values
    
    # Color scheme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    print("\n📊 Creating separate visualizations for better readability...\n")
    
    # ============= PLOT 1: Final Loss Comparison =============
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(range(len(optimizers)), losses, yerr=stds, 
                   capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yscale('log')
    plt.xticks(range(len(optimizers)), optimizers, rotation=0, fontsize=12, fontweight='bold')
    plt.ylabel('Final Loss (log scale)', fontsize=12, fontweight='bold')
    plt.title('Final Loss Comparison with Error Bars\n(Lower is Better)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (loss, std) in enumerate(zip(losses, stds)):
        plt.text(i, loss, f'{loss:.2e}\n±{std:.2e}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '01_final_loss_comparison.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✅ 1/6: Final Loss Comparison saved to {output_file}")
    plt.close()
    
    # ============= PLOT 2: Distance to Optimum =============
    plt.figure(figsize=(10, 6))
    
    # Calculate std from detailed data
    dist_stds = [detailed_df[detailed_df['optimizer'] == opt]['distance_to_optimum'].std() 
                 for opt in optimizers]
    
    bars = plt.bar(range(len(optimizers)), distances, yerr=dist_stds, 
                   capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xticks(range(len(optimizers)), optimizers, rotation=0, fontsize=12, fontweight='bold')
    plt.ylabel('Distance to Optimum (1,1)', fontsize=12, fontweight='bold')
    plt.title('Distance to Global Optimum\n(Lower is Better)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (dist, std) in enumerate(zip(distances, dist_stds)):
        plt.text(i, dist, f'{dist:.4f}\n±{std:.4f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '02_distance_to_optimum.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✅ 2/6: Distance to Optimum saved to {output_file}")
    plt.close()
    
    # ============= PLOT 3: Convergence Success Rate =============
    plt.figure(figsize=(10, 6))
    
    # Parse convergence rate from string "X/5"
    conv_rates = [int(conv.split('/')[0])/int(conv.split('/')[1])*100 
                  for conv in summary_df['Converged'].values]
    
    bars = plt.bar(range(len(optimizers)), conv_rates, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xticks(range(len(optimizers)), optimizers, rotation=0, fontsize=12, fontweight='bold')
    plt.ylabel('Convergence Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Convergence Success Rate (5 seeds)\n(Higher is Better)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, rate in enumerate(conv_rates):
        plt.text(i, rate + 3, f'{rate:.0f}%', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '03_convergence_rate.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✅ 3/6: Convergence Rate saved to {output_file}")
    plt.close()
    
    # ============= PLOT 4: Box Plot Distribution =============
    plt.figure(figsize=(10, 6))
    
    box_data = [detailed_df[detailed_df['optimizer'] == opt]['final_loss'].values 
                for opt in optimizers]
    
    bp = plt.boxplot(box_data, tick_labels=optimizers, patch_artist=True, 
                     showmeans=True, meanline=True,
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.yscale('log')
    plt.ylabel('Final Loss (log scale)', fontsize=12, fontweight='bold')
    plt.xlabel('Optimizer', fontsize=12, fontweight='bold')
    plt.title('Loss Distribution Across Seeds\n(Red=Median, Blue=Mean)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '04_loss_distribution_boxplot.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✅ 4/6: Loss Distribution Box Plot saved to {output_file}")
    plt.close()
    
    # ============= PLOT 5: Statistical Significance Heatmap =============
    plt.figure(figsize=(10, 8))
    
    # Create p-value matrix - parse comparison strings
    p_matrix = np.ones((len(optimizers), len(optimizers)))
    for _, row in stats_df.iterrows():
        comp_parts = row['Comparison'].split(' vs ')
        opt1 = comp_parts[0]
        opt2 = comp_parts[1]
        opt1_idx = optimizers.tolist().index(opt1)
        opt2_idx = optimizers.tolist().index(opt2)
        p_matrix[opt1_idx, opt2_idx] = row['p-value']
        p_matrix[opt2_idx, opt1_idx] = row['p-value']
    
    # Create mask for diagonal
    mask = np.eye(len(optimizers), dtype=bool)
    
    # Plot heatmap
    sns.heatmap(p_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r',
                xticklabels=optimizers, yticklabels=optimizers,
                cbar_kws={'label': 'p-value'}, vmin=0, vmax=0.05,
                mask=mask, linewidths=2, linecolor='black')
    
    plt.title('Statistical Significance Matrix (p-values)\nGreen = Significant (p<0.05), Red = Not Significant', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Optimizer', fontsize=12, fontweight='bold')
    plt.ylabel('Optimizer', fontsize=12, fontweight='bold')
    
    # Add significance annotations
    for i in range(len(optimizers)):
        for j in range(len(optimizers)):
            if i != j and not np.isnan(p_matrix[i, j]) and p_matrix[i, j] != 1.0:
                p_val = p_matrix[i, j]
                if p_val < 0.001:
                    symbol = '***'
                elif p_val < 0.01:
                    symbol = '**'
                elif p_val < 0.05:
                    symbol = '*'
                else:
                    symbol = 'ns'
                plt.text(j+0.5, i+0.7, symbol, ha='center', va='center',
                        fontsize=16, fontweight='bold', color='white')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '05_statistical_significance_heatmap.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✅ 5/6: Statistical Significance Heatmap saved to {output_file}")
    plt.close()
    
    # ============= PLOT 6: Effect Sizes =============
    plt.figure(figsize=(12, 6))
    
    comparisons = [row['Comparison'].replace(' vs ', '\nvs\n') 
                   for _, row in stats_df.iterrows()]
    effect_sizes = stats_df['Cohens d'].values
    
    # Color based on effect size magnitude
    bar_colors = []
    for d in effect_sizes:
        abs_d = abs(d)
        if abs_d < 0.2:
            bar_colors.append('#95E1D3')  # Small
        elif abs_d < 0.5:
            bar_colors.append('#F38181')  # Medium
        elif abs_d < 0.8:
            bar_colors.append('#AA96DA')  # Large
        else:
            bar_colors.append('#FCBAD3')  # Very large
    
    bars = plt.bar(range(len(comparisons)), effect_sizes, 
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small (0.2)')
    plt.axhline(y=-0.2, color='green', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
    plt.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
    plt.axhline(y=-0.8, color='red', linestyle='--', alpha=0.5)
    
    plt.xticks(range(len(comparisons)), comparisons, rotation=0, fontsize=10)
    plt.ylabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
    plt.title("Effect Sizes for Pairwise Comparisons\n(Negative = First optimizer is better)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (d, effect) in enumerate(zip(effect_sizes, stats_df['Effect'].values)):
        y_pos = d + (0.3 if d > 0 else -0.3)
        plt.text(i, y_pos, f'd={d:.2f}\n({effect})', 
                 ha='center', va='bottom' if d > 0 else 'top', 
                 fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '06_effect_sizes.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✅ 6/6: Effect Sizes saved to {output_file}")
    plt.close()
    
    print("\n" + "="*60)
    print("🎉 SUCCESS! All 6 separate visualizations created:")
    print("="*60)
    for i in range(1, 7):
        print(f"📊 {output_dir}/0{i}_*.png")
    print("="*60)
    print("\n✅ Each plot is high-resolution (300 DPI) and optimized!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create separate high-quality plots from experiment results'
    )
    parser.add_argument('--summary', type=str, default='results/optimizer_summary.csv',
                        help='Path to optimizer summary CSV')
    parser.add_argument('--stats', type=str, default='results/statistical_comparisons.csv',
                        help='Path to statistical comparisons CSV')
    parser.add_argument('--detailed', type=str, default='results/multiseed_detailed.csv',
                        help='Path to detailed results CSV')
    parser.add_argument('--output', type=str, default='plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    create_separate_plots(
        summary_csv=args.summary,
        stats_csv=args.stats,
        detailed_csv=args.detailed,
        output_dir=args.output
    )
