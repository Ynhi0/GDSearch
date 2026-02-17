"""
Module defining test functions for evaluating optimization algorithms.
Includes helper function to get test function instances by name.
"""

# This file extends test_functions.py with a factory function

from src.core.test_functions import (
   Rosenbrock, IllConditionedQuadratic, SaddlePoint, Ackley2D  
)


def get_test_function(name: str):
    """
    Get test function instance by name.
    
    Args:
        name: Function name ('rosenbrock', 'ill_conditioned_quadratic', 'saddle_point', 'ackley2d')
        
    Returns:
        TestFunction instance
        
    Raises:
        ValueError: If function name is not recognized
    """
    name = name.lower().replace('-', '_').replace(' ', '_')
    
    if name == 'rosenbrock':
        return Rosenbrock()
    elif name in ['ill_conditioned_quadratic', 'illconditionedquadratic', 'quadratic']:
        return IllConditionedQuadratic()
    elif name in ['saddle_point', 'saddlepoint', 'saddle']:
        return SaddlePoint()
    elif name in ['ackley2d', 'ackley_2d']:
        return Ackley2D()
    else:
        raise ValueError(
            f"Unknown test function: '{name}'. "
            f"Available: rosenbrock, ill_conditioned_quadratic, saddle_point, ackley2d"
        )
