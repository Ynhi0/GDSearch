"""
Visualization utilities for Hessian eigenvalue analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_eigenvalue_evolution(df: pd.DataFrame, 
                               title: str = "Hessian Eigenvalue Evolution",
                               save_path: Optional[str] = None):
    """
    Plot the evolution of Hessian eigenvalues along optimization trajectory.
    
    Args:
        df: DataFrame with columns ['iteration', 'lambda_min', 'lambda_max', 'condition_number']
        title: Plot title
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot lambda_max and lambda_min
    ax1 = axes[0]
    ax1.plot(df['iteration'], df['lambda_max'], label='λ_max (largest eigenvalue)', 
             color='red', linewidth=1.5, alpha=0.8)
    ax1.plot(df['iteration'], df['lambda_min'], label='λ_min (smallest eigenvalue)', 
             color='blue', linewidth=1.5, alpha=0.8)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title(f'{title}\nHessian Eigenvalues')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot condition number (log scale)
    ax2 = axes[1]
    # Filter out infinite values for plotting
    cond_finite = df['condition_number'].replace([np.inf, -np.inf], np.nan)
    ax2.semilogy(df['iteration'], cond_finite, color='purple', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Condition Number (log scale)')
    ax2.set_title('Local Conditioning (κ = |λ_max / λ_min|)')
    ax2.grid(True, alpha=0.3)
    
    # Plot eigenvalue product (sign indicates saddle vs minimum)
    ax3 = axes[2]
    product = df['lambda_min'] * df['lambda_max']
    colors = ['green' if p > 0 else 'red' for p in product]
    ax3.scatter(df['iteration'], product, c=colors, s=1, alpha=0.5)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_ylabel('λ_min × λ_max')
    ax3.set_xlabel('Iteration')
    ax3.set_title('Eigenvalue Product (>0: locally convex, <0: saddle point)')
    ax3.grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Convex (both eigenvalues same sign)'),
        Patch(facecolor='red', label='Saddle (eigenvalues opposite signs)')
    ]
    ax3.legend(handles=legend_elements, loc='best', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenvalue evolution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_eigenvalue_trajectory_overlay(df: pd.DataFrame,
                                       test_function,
                                       title: str = "Trajectory with Curvature Overlay",
                                       save_path: Optional[str] = None):
    """
    Plot optimization trajectory with color-coded condition number.
    
    Args:
        df: DataFrame with columns ['x', 'y', 'condition_number']
        test_function: Test function object for plotting contours
        title: Plot title
        save_path: If provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot function contours
    x_bounds, y_bounds = test_function.get_bounds()
    x = np.linspace(x_bounds[0], x_bounds[1], 100)
    y = np.linspace(y_bounds[0], y_bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[test_function.compute(xi, yi) for xi in x] for yi in y])
    
    # Use log scale for better visualization
    levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max() + 1), 20)
    contour = ax.contour(X, Y, Z, levels=levels, alpha=0.3, colors='gray')
    
    # Plot trajectory colored by condition number
    # Use log scale for condition number coloring
    cond_finite = df['condition_number'].replace([np.inf, -np.inf], np.nan)
    log_cond = np.log10(cond_finite + 1)  # +1 to avoid log(0)
    
    scatter = ax.scatter(df['x'], df['y'], c=log_cond, cmap='coolwarm', 
                        s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add trajectory line
    ax.plot(df['x'], df['y'], 'k-', alpha=0.3, linewidth=0.5)
    
    # Mark start and end
    ax.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', markersize=10, 
            label='Start', markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'r*', markersize=15, 
            label='End', markeredgecolor='black', markeredgewidth=1.5)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(Condition Number + 1)', rotation=270, labelpad=20)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title}\n(Color shows local conditioning)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory overlay plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_saddle_escape(df: pd.DataFrame, 
                          saddle_threshold: float = 0.1,
                          min_iterations: int = 10) -> dict:
    """
    Analyze how quickly the optimizer escapes saddle point regions.
    
    A saddle point region is defined where λ_min × λ_max < 0 and |λ_min| is small.
    
    Args:
        df: DataFrame with eigenvalue columns
        saddle_threshold: Threshold for considering a point near saddle
        min_iterations: Minimum iterations to count as "stuck"
    
    Returns:
        Dictionary with analysis results
    """
    product = df['lambda_min'] * df['lambda_max']
    near_saddle = product < 0  # Opposite sign eigenvalues
    small_curvature = np.abs(df['lambda_min']) < saddle_threshold
    
    in_saddle_region = near_saddle & small_curvature
    
    # Find consecutive saddle regions
    saddle_regions = []
    in_region = False
    start_iter = 0
    
    for i, is_saddle in enumerate(in_saddle_region):
        if is_saddle and not in_region:
            start_iter = i
            in_region = True
        elif not is_saddle and in_region:
            duration = i - start_iter
            if duration >= min_iterations:
                saddle_regions.append({
                    'start': start_iter,
                    'end': i,
                    'duration': duration
                })
            in_region = False
    
    # Close final region if needed
    if in_region:
        duration = len(df) - start_iter
        if duration >= min_iterations:
            saddle_regions.append({
                'start': start_iter,
                'end': len(df),
                'duration': duration
            })
    
    total_saddle_iters = sum(r['duration'] for r in saddle_regions)
    
    return {
        'num_saddle_regions': len(saddle_regions),
        'saddle_regions': saddle_regions,
        'total_iterations_in_saddle': total_saddle_iters,
        'percent_time_in_saddle': 100 * total_saddle_iters / len(df) if len(df) > 0 else 0
    }


if __name__ == '__main__':
    # Demo: load a result CSV and plot eigenvalue evolution
    import sys
    import glob
    
    csv_files = glob.glob('results/*.csv')
    if not csv_files:
        print("No CSV files found in results/")
        sys.exit(1)
    
    # Filter for 2D experiments (not NN)
    gd_files = [f for f in csv_files if not f.startswith('results/NN_')]
    
    if not gd_files:
        print("No 2D experiment CSV files found")
        sys.exit(1)
    
    print(f"Found {len(gd_files)} 2D experiment files")
    print("Plotting eigenvalue evolution for first 3 files...")
    
    for csv_path in gd_files[:3]:
        df = pd.read_csv(csv_path)
        
        # Check if eigenvalue columns exist
        if 'lambda_min' not in df.columns:
            print(f"⚠️  {csv_path} does not have eigenvalue data (old format)")
            continue
        
        basename = csv_path.replace('results/', '').replace('.csv', '')
        plot_path = f'plots/{basename}_eigenvalues.png'
        
        plot_eigenvalue_evolution(df, title=f"Eigenvalues: {basename}", 
                                 save_path=plot_path)
        
        # Saddle escape analysis
        analysis = analyze_saddle_escape(df)
        print(f"\n{basename}:")
        print(f"  Saddle regions detected: {analysis['num_saddle_regions']}")
        print(f"  Time in saddle: {analysis['percent_time_in_saddle']:.1f}%")
