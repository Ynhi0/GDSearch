"""Experiment runners module.

Modular experiment runners extracted from run_all_kaggle.py.
"""

# Note: Runners are wrapper modules that delegate to run_all_kaggle implementations
# They provide a cleaner interface using ExperimentConfig

__all__ = []

# Import runners if available
try:
    from . import mnist_runner
    __all__.append('mnist_runner')
except ImportError:
    pass

try:
    from . import cifar10_runner
    __all__.append('cifar10_runner')
except ImportError:
    pass
