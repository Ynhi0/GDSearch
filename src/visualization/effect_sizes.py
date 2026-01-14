"""
Effect Size Visualization Module.

Creates forest plots, heatmaps, and distribution plots for effect sizes.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd


def plot_effect_sizes_forest(stat_results: List[Dict[str, Any]], 
                             save_path: Path,
                             title: str = "Optimizer Effect Sizes with 95% Confidence Intervals") -> None:
    """
    Create forest plot of effect sizes with confidence intervals.
    
    Args:
        stat_results: List of statistical comparison results
        save_path: Path to save the plot
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        if not stat_results:
            logging.warning("No statistical results to plot")
            return
        
        # Prepare data
        comparisons = []
        cohens_ds = []
        ci_lowers = []
        ci_uppers = []
        significants = []
        
        for result in stat_results:
            comp = f"{result['optimizer_a']} vs {result['optimizer_b']}"
            d = result.get('cohens_d', np.nan)
            ci_low = result.get('cohens_d_ci_lower', np.nan)
            ci_high = result.get('cohens_d_ci_upper', np.nan)
            sig = result.get('significant', False)
            
            if not np.isnan(d) and not np.isnan(ci_low) and not np.isnan(ci_high):
                comparisons.append(comp)
                cohens_ds.append(d)
                ci_lowers.append(d - ci_low)
                ci_uppers.append(ci_high - d)
                significants.append(sig)
        
        if not comparisons:
            logging.warning("No valid effect sizes with CIs to plot")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(comparisons) * 0.5)))
        
        y_positions = list(range(len(comparisons)))
        
        # Plot error bars and points individually for each comparison
        for i, (d, ci_low, ci_up, sig) in enumerate(zip(cohens_ds, ci_lowers, ci_uppers, significants)):
            color = 'red' if sig else 'gray'
            ax.errorbar(d, i, xerr=[[ci_low], [ci_up]],
                       fmt='o', color='none', ecolor=color,
                       capsize=5, capthick=2, markersize=8)
            ax.plot(d, i, 'o', color=color, markersize=8)
        
        # Plot points
        for i, (d, sig) in enumerate(zip(cohens_ds, significants)):
            ax.plot(d, i, 'o', color='red' if sig else 'gray', markersize=8)
        
        # Add reference lines
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(-0.2, color='blue', linestyle=':', linewidth=0.8, alpha=0.3)
        ax.axvline(0.2, color='blue', linestyle=':', linewidth=0.8, alpha=0.3)
        ax.axvline(-0.5, color='orange', linestyle=':', linewidth=0.8, alpha=0.3)
        ax.axvline(0.5, color='orange', linestyle=':', linewidth=0.8, alpha=0.3)
        ax.axvline(-0.8, color='red', linestyle=':', linewidth=0.8, alpha=0.3)
        ax.axvline(0.8, color='red', linestyle=':', linewidth=0.8, alpha=0.3)
        
        # Labels and formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(comparisons)
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add effect size thresholds legend
        threshold_text = (
            "Effect Size Thresholds:\n"
            "  |d| < 0.2: negligible\n"
            "  0.2 ≤ |d| < 0.5: small\n"
            "  0.5 ≤ |d| < 0.8: medium\n"
            "  |d| ≥ 0.8: large"
        )
        ax.text(0.02, 0.98, threshold_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Forest plot saved to {save_path}")
    except ImportError:
        logging.warning("matplotlib not available, cannot create forest plot")
    except Exception as e:
        logging.error(f"Failed to create forest plot: {e}")


def plot_effect_sizes_heatmap(stat_results: List[Dict[str, Any]], 
                              save_path: Path,
                              metric: str = 'cohens_d') -> None:
    """
    Create heatmap of pairwise effect sizes.
    
    Args:
        stat_results: List of statistical comparison results
        save_path: Path to save the plot
        metric: Metric to plot ('cohens_d', 'p_value', 'p_value_adjusted')
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not stat_results:
            logging.warning("No statistical results to plot")
            return
        
        # Get all unique optimizers
        optimizers = set()
        for result in stat_results:
            optimizers.add(result['optimizer_a'])
            optimizers.add(result['optimizer_b'])
        optimizers = sorted(list(optimizers))
        
        # Create matrix
        n = len(optimizers)
        matrix = np.zeros((n, n))
        matrix[:] = np.nan
        
        opt_to_idx = {opt: i for i, opt in enumerate(optimizers)}
        
        for result in stat_results:
            i = opt_to_idx[result['optimizer_a']]
            j = opt_to_idx[result['optimizer_b']]
            value = result.get(metric, np.nan)
            
            if not np.isnan(value):
                matrix[i, j] = value
                matrix[j, i] = -value if metric == 'cohens_d' else value
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))
        
        if metric == 'cohens_d':
            cmap = 'RdBu_r'
            center = 0
            vmin, vmax = -2, 2
        else:
            cmap = 'RdYlGn_r'
            center = 0.05
            vmin, vmax = 0, 1
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, center=center,
                   xticklabels=optimizers, yticklabels=optimizers,
                   vmin=vmin, vmax=vmax, square=True, linewidths=0.5,
                   cbar_kws={'label': metric.replace('_', ' ').title()},
                   ax=ax)
        
        ax.set_title(f'Pairwise {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Heatmap saved to {save_path}")
    except ImportError as e:
        logging.warning(f"Required library not available for heatmap: {e}")
    except Exception as e:
        logging.error(f"Failed to create heatmap: {e}")


def plot_effect_size_distribution(stat_results: List[Dict[str, Any]], 
                                  save_path: Path) -> None:
    """
    Create distribution plot of effect sizes.
    
    Args:
        stat_results: List of statistical comparison results
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        if not stat_results:
            logging.warning("No statistical results to plot")
            return
        
        # Extract effect sizes
        effect_sizes = [r['cohens_d'] for r in stat_results 
                       if not np.isnan(r.get('cohens_d', np.nan))]
        
        if not effect_sizes:
            logging.warning("No valid effect sizes to plot")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(effect_sizes, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='No effect')
        ax1.axvline(-0.2, color='blue', linestyle=':', linewidth=1, alpha=0.5, label='Small threshold')
        ax1.axvline(0.2, color='blue', linestyle=':', linewidth=1, alpha=0.5)
        ax1.axvline(-0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Medium threshold')
        ax1.axvline(0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax1.set_xlabel("Cohen's d", fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Effect Sizes', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(effect_sizes, vert=True, widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='steelblue'),
                   whiskerprops=dict(color='steelblue'),
                   capprops=dict(color='steelblue'),
                   medianprops=dict(color='red', linewidth=2))
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(-0.2, color='blue', linestyle=':', linewidth=1, alpha=0.3)
        ax2.axhline(0.2, color='blue', linestyle=':', linewidth=1, alpha=0.3)
        ax2.axhline(-0.5, color='orange', linestyle=':', linewidth=1, alpha=0.3)
        ax2.axhline(0.5, color='orange', linestyle=':', linewidth=1, alpha=0.3)
        ax2.set_ylabel("Cohen's d", fontsize=12)
        ax2.set_title('Effect Size Summary', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(['All Comparisons'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = (
            f"n = {len(effect_sizes)}\n"
            f"Mean = {np.mean(effect_sizes):.3f}\n"
            f"Median = {np.median(effect_sizes):.3f}\n"
            f"Std = {np.std(effect_sizes):.3f}"
        )
        ax2.text(0.5, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Distribution plot saved to {save_path}")
    except ImportError:
        logging.warning("matplotlib not available, cannot create distribution plot")
    except Exception as e:
        logging.error(f"Failed to create distribution plot: {e}")


def create_all_effect_size_plots(stat_results: List[Dict[str, Any]], 
                                 output_dir: Path) -> None:
    """
    Create all effect size visualization plots.
    
    Args:
        stat_results: List of statistical comparison results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logging.info("Creating effect size visualizations...")
    
    # Forest plot
    plot_effect_sizes_forest(stat_results, 
                             output_dir / "effect_sizes_forest_plot.png")
    
    # Heatmaps
    plot_effect_sizes_heatmap(stat_results, 
                             output_dir / "effect_sizes_heatmap.png",
                             metric='cohens_d')
    
    plot_effect_sizes_heatmap(stat_results, 
                             output_dir / "pvalues_heatmap.png",
                             metric='p_value_adjusted')
    
    # Distribution
    plot_effect_size_distribution(stat_results, 
                                  output_dir / "effect_sizes_distribution.png")
    
    logging.info(f"All effect size plots saved to {output_dir}")
