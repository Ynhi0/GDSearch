"""
Visualization: plotting results, eigenvalues, loss landscapes.
"""

# Re-export commonly used plotting helpers for convenience
from .loss_landscape import plot_loss_landscape, create_loss_landscape_animation
from .plot_results import plot_step_size_vs_iteration, plot_trajectory_and_step_size

__all__ = [
    'plot_trajectory',
    'plot_metrics',
    'plot_comparison',
    'plot_multiseed_comparison',
    'plot_final_metric_comparison',
    'plot_loss_landscape_2d',
    'plot_loss_landscape',
    'create_loss_landscape_animation',
    'plot_step_size_vs_iteration',
    'plot_trajectory_and_step_size',
    'plot_eigenvalue_evolution',
]
