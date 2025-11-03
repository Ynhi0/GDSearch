"""
GDSearch: Gradient Descent Optimization Research Framework

A comprehensive framework for comparing gradient descent algorithms on
2D test functions and neural networks with rigorous statistical analysis.
"""

__version__ = '2.0.0'
__author__ = 'GDSearch Team'

# Make core modules easily accessible
from src import core, experiments, analysis, visualization

__all__ = ['core', 'experiments', 'analysis', 'visualization']
