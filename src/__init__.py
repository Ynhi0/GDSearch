"""
GDSearch: Gradient Descent Optimization Research Framework

A comprehensive framework for comparing gradient descent algorithms on
2D test functions and neural networks with rigorous statistical analysis.
"""

__version__ = '2.0.0'
__author__ = 'GDSearch Team'

# Explicitly import subpackages where available to ensure `from src import core` works
__all__ = []
try:
    from . import core, experiments, analysis, visualization  # noqa: F401
    __all__.extend(['core', 'experiments', 'analysis', 'visualization'])
except Exception as e:
    import logging
    logging.debug("Could not import subpackages in src package __init__: %s", e, exc_info=True)
    # Keep import-safe: optional subpackages may be unavailable in constrained envs
