"""
Anti-Aliasing Visualization Utilities

This module provides robust plotting utilities that avoid aliasing artifacts
when downsampling high-frequency data for visualization.

Addresses QA Issue #14: "Visualization Aliasing"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Tuple, List, Union
from scipy.ndimage import uniform_filter1d


def rolling_mean_with_bounds(
    data: np.ndarray,
    window_size: int = 10,
    compute_bounds: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute rolling mean with min/max bounds to avoid aliasing.

    When downsampling oscillatory data (e.g., every 10th point), you can
    miss peaks and troughs (aliasing). Using a rolling mean + shaded bounds
    preserves the envelope of the signal.

    Args:
        data: 1D array of values to smooth
        window_size: Size of rolling window
        compute_bounds: If True, compute min/max within each window

    Returns:
        mean: Rolling mean
        lower_bound: Min value within each window (if compute_bounds=True)
        upper_bound: Max value within each window (if compute_bounds=True)
    """
    if len(data) == 0:
        return np.array([]), None, None

    # Compute rolling mean using scipy for speed
    mean = uniform_filter1d(data, size=window_size, mode='nearest')

    if not compute_bounds:
        return mean, None, None

    # Compute rolling min/max
    half_window = window_size // 2
    lower_bound = np.zeros_like(data)
    upper_bound = np.zeros_like(data)

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        lower_bound[i] = np.min(window)
        upper_bound[i] = np.max(window)

    return mean, lower_bound, upper_bound


def downsample_with_envelope(
    x: np.ndarray,
    y: np.ndarray,
    factor: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample data while preserving envelope (min/max) information.

    Instead of just taking every Nth point (which causes aliasing),
    this computes the mean, min, and max within each downsampling window.

    Args:
        x: X-axis values (e.g., iterations)
        y: Y-axis values (e.g., loss)
        factor: Downsampling factor (e.g., 10 = keep every 10th window)

    Returns:
        x_down: Downsampled x values (window centers)
        y_mean: Mean y value in each window
        y_min: Min y value in each window
        y_max: Max y value in each window
    """
    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    n_windows = len(x) // factor
    if n_windows == 0:
        # Data too short to downsample
        return x, y, y, y

    x_down = np.zeros(n_windows)
    y_mean = np.zeros(n_windows)
    y_min = np.zeros(n_windows)
    y_max = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * factor
        end = min((i + 1) * factor, len(x))

        x_down[i] = np.mean(x[start:end])
        y_mean[i] = np.mean(y[start:end])
        y_min[i] = np.min(y[start:end])
        y_max[i] = np.max(y[start:end])

    return x_down, y_mean, y_min, y_max


def plot_with_envelope(
    x: np.ndarray,
    y: np.ndarray,
    ax: Optional[Axes] = None,
    label: str = '',
    color: Optional[str] = None,
    downsample_factor: Optional[int] = None,
    rolling_window: Optional[int] = None,
    show_bounds: bool = True,
    alpha_line: float = 0.8,
    alpha_fill: float = 0.2
) -> Axes:
    """
    Plot line with envelope (min/max bounds) to avoid aliasing artifacts.

    This is the CORRECT way to plot high-frequency training curves.

    Usage modes:
    1. Downsampling mode: Set downsample_factor to reduce data density
    2. Smoothing mode: Set rolling_window to apply rolling mean
    3. Both: Apply both transformations

    Args:
        x: X-axis values
        y: Y-axis values
        ax: Matplotlib axes (creates new if None)
        label: Legend label
        color: Line color
        downsample_factor: If set, downsample data by this factor
        rolling_window: If set, apply rolling mean with this window size
        show_bounds: Whether to show min/max envelope
        alpha_line: Transparency of main line
        alpha_fill: Transparency of envelope fill

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Handle empty data
    if len(x) == 0 or len(y) == 0:
        return ax

    x_plot = x.copy()
    y_plot = y.copy()
    y_lower = None
    y_upper = None

    # Step 1: Apply downsampling if requested
    if downsample_factor is not None and downsample_factor > 1:
        x_plot, y_plot, y_lower, y_upper = downsample_with_envelope(
            x_plot, y_plot, factor=downsample_factor
        )

    # Step 2: Apply rolling mean if requested
    if rolling_window is not None and rolling_window > 1:
        y_smooth, y_lower_smooth, y_upper_smooth = rolling_mean_with_bounds(
            y_plot, window_size=rolling_window, compute_bounds=show_bounds
        )
        y_plot = y_smooth
        if show_bounds and y_lower_smooth is not None:
            y_lower = y_lower_smooth
            y_upper = y_upper_smooth

    # Plot main line
    ax.plot(x_plot, y_plot, label=label, color=color, alpha=alpha_line, linewidth=1.5)

    # Plot envelope (shaded region)
    if show_bounds and y_lower is not None and y_upper is not None:
        ax.fill_between(
            x_plot,
            y_lower,
            y_upper,
            color=color,
            alpha=alpha_fill,
            linewidth=0
        )

    return ax


def plot_multiple_runs_with_statistics(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    ax: Optional[Axes] = None,
    downsample_factor: Optional[int] = None,
    rolling_window: Optional[int] = None,
    show_individual_runs: bool = False,
    alpha_individual: float = 0.15
) -> Axes:
    """
    Plot multiple runs with mean ± std envelope.

    This is the GOLD STANDARD for plotting multi-seed experiments:
    - Shows mean trajectory (central tendency)
    - Shows std envelope (variability)
    - Optionally shows individual runs (faint lines)
    - Avoids aliasing through downsampling + smoothing

    Args:
        data: DataFrame with columns [x_col, y_col, group_col, seed_col]
        x_col: Column name for x-axis (e.g., 'iteration')
        y_col: Column name for y-axis (e.g., 'loss')
        group_col: Column name for grouping (e.g., 'optimizer')
        ax: Matplotlib axes
        downsample_factor: Downsampling factor
        rolling_window: Rolling mean window size
        show_individual_runs: Whether to show individual seed runs
        alpha_individual: Transparency for individual runs

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    # Iterate over groups (e.g., optimizers)
    for group_name, group_data in data.groupby(group_col):
        # Compute statistics across seeds at each x value
        stats = group_data.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()

        x_vals = stats[x_col].values  # type: ignore[assignment]
        y_mean = stats['mean'].values  # type: ignore[assignment]
        y_std = stats['std'].values  # type: ignore[assignment]

        # Apply transformations
        if downsample_factor is not None and downsample_factor > 1:
            x_vals, y_mean, _, _ = downsample_with_envelope(x_vals, y_mean, downsample_factor)  # type: ignore[arg-type]
            _, y_std, _, _ = downsample_with_envelope(stats[x_col].values, y_std, downsample_factor)  # type: ignore[arg-type]

        if rolling_window is not None and rolling_window > 1:
            y_mean, _, _ = rolling_mean_with_bounds(y_mean, rolling_window, compute_bounds=False)  # type: ignore[arg-type]
            y_std, _, _ = rolling_mean_with_bounds(y_std, rolling_window, compute_bounds=False)  # type: ignore[arg-type]

        # Plot mean line
        ax.plot(x_vals, y_mean, label=str(group_name), linewidth=2)  # type: ignore[arg-type]

        # Plot std envelope
        ax.fill_between(  # type: ignore[arg-type]
            np.asarray(x_vals),
            y_mean - y_std,  # type: ignore[operator]
            y_mean + y_std,  # type: ignore[operator]
            alpha=0.25,
            linewidth=0
        )

        # Optionally plot individual runs
        if show_individual_runs and 'seed' in group_data.columns:
            for seed, seed_data in group_data.groupby('seed'):
                x_seed = seed_data[x_col].values  # type: ignore[assignment]
                y_seed = seed_data[y_col].values  # type: ignore[assignment]

                if downsample_factor is not None:
                    x_seed, y_seed, _, _ = downsample_with_envelope(x_seed, y_seed, downsample_factor)  # type: ignore[arg-type]

                ax.plot(x_seed, y_seed, alpha=alpha_individual, linewidth=0.5)  # type: ignore[arg-type]

    return ax


def detect_oscillation_frequency(y: np.ndarray) -> float:
    """
    Detect dominant oscillation frequency using FFT.

    This helps determine appropriate downsampling/smoothing parameters.

    Args:
        y: Signal to analyze

    Returns:
        dominant_frequency: Dominant frequency (cycles per sample)
    """
    if len(y) < 10:
        return 0.0

    # Remove DC component
    y_centered = y - np.mean(y)

    # Compute FFT
    fft = np.fft.rfft(y_centered)
    freqs = np.fft.rfftfreq(len(y_centered))

    # Find dominant frequency (excluding DC)
    power = np.abs(fft[1:]) ** 2
    if len(power) == 0:
        return 0.0

    dominant_idx = np.argmax(power) + 1
    dominant_freq = freqs[dominant_idx]

    return float(dominant_freq)


def recommend_smoothing_params(
    y: np.ndarray,
    target_points: int = 500
) -> Tuple[int, int]:
    """
    Recommend downsampling and smoothing parameters based on data characteristics.

    This automates the selection of visualization parameters to avoid aliasing
    while maintaining visual clarity.

    Args:
        y: Signal to analyze
        target_points: Target number of points for visualization

    Returns:
        downsample_factor: Recommended downsampling factor
        rolling_window: Recommended rolling window size
    """
    n_points = len(y)

    # Compute downsampling factor
    if n_points <= target_points:
        downsample_factor = 1
    else:
        downsample_factor = max(1, n_points // target_points)

    # Detect oscillation frequency
    dominant_freq = detect_oscillation_frequency(y)

    # Nyquist criterion: need at least 2 samples per cycle
    # If downsampling, ensure we don't alias
    if dominant_freq > 0:
        nyquist_factor = int(np.ceil(1.0 / (2 * dominant_freq)))
        downsample_factor = min(downsample_factor, nyquist_factor)

    # Recommend rolling window (roughly 5-10% of downsampled length)
    effective_length = n_points // max(1, downsample_factor)
    rolling_window = max(1, effective_length // 20)

    return int(downsample_factor), int(rolling_window)


# Example usage function
def example_usage():
    """
    Demonstrate proper anti-aliasing visualization.
    """
    # Generate synthetic oscillatory data
    np.random.seed(42)
    x = np.arange(10000)
    y = np.sin(x * 0.01) + 0.1 * np.random.randn(10000)

    # Get recommendations
    downsample, window = recommend_smoothing_params(y)
    print(f"Recommended: downsample={downsample}, window={window}")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # WRONG: Naive downsampling (aliasing)
    axes[0, 0].plot(x[::100], y[::100], 'o-', markersize=2)
    axes[0, 0].set_title('❌ WRONG: Naive Downsampling (Aliasing)', fontweight='bold', color='red')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')

    # CORRECT: Envelope-based downsampling
    plot_with_envelope(x, y, ax=axes[0, 1], downsample_factor=100, show_bounds=True)
    axes[0, 1].set_title('✅ CORRECT: Envelope Downsampling', fontweight='bold', color='green')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')

    # CORRECT: Rolling mean
    plot_with_envelope(x, y, ax=axes[1, 0], rolling_window=50, show_bounds=True)
    axes[1, 0].set_title('✅ CORRECT: Rolling Mean + Bounds', fontweight='bold', color='green')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')

    # BEST: Downsample + Rolling mean
    plot_with_envelope(x, y, ax=axes[1, 1], downsample_factor=10, rolling_window=10, show_bounds=True)
    axes[1, 1].set_title('✅ BEST: Downsample + Smooth + Envelope', fontweight='bold', color='darkgreen')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig('antialiasing_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison to antialiasing_comparison.png")


if __name__ == '__main__':
    example_usage()
