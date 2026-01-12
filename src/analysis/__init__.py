"""
Analysis tools: statistics, ablation, baseline comparison, sensitivity.
"""

__all__ = []
# Import and expose commonly-used analysis functions if present
try:
    from .statistical_analysis import compare_optimizers_ttest, print_ttest_results  # noqa: F401
    __all__.extend(['compare_optimizers_ttest', 'print_ttest_results'])
except Exception as e:
    import logging
    logging.debug("Optional import statistical_analysis failed: %s", e, exc_info=True)

try:
    from .ablation_study import run_ablation_study  # noqa: F401
    __all__.append('run_ablation_study')
except Exception as e:
    import logging
    logging.debug("Optional import ablation_study failed: %s", e, exc_info=True)

# Additional analysis utilities may be added here; avoid listing names not present to satisfy static checks.
