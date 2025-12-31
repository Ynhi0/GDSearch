"""
Visualization: plotting results, eigenvalues, loss landscapes.
"""

__all__ = []
# Import commonly used plotting utilities with fallbacks for import-safety
try:
    from .plot_results import (
        plot_trajectory, plot_metrics, plot_comparison, plot_multiseed_comparison,
        plot_final_metric_comparison, plot_step_size_vs_iteration, plot_trajectory_and_step_size
    )  # noqa: F401
    __all__.extend([
        'plot_trajectory', 'plot_metrics', 'plot_comparison', 'plot_multiseed_comparison',
        'plot_final_metric_comparison', 'plot_step_size_vs_iteration', 'plot_trajectory_and_step_size'
    ])
except Exception:
    pass

try:
    from .loss_landscape import plot_loss_landscape, create_loss_landscape_animation  # noqa: F401
    from .plot_eigenvalues import plot_eigenvalue_evolution  # noqa: F401
    __all__.extend(['plot_loss_landscape', 'create_loss_landscape_animation', 'plot_eigenvalue_evolution'])
except Exception:
    pass
