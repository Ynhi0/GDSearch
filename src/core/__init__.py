"""
Core implementations: optimizers, test functions, models, data utilities.
"""

from .optimizers import SGD, SGDMomentum, RMSProp, Adam, AdamW
from .test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint
from .models import SimpleMLP, SimpleCNN, ConvNet
from .data_utils import get_mnist_loaders, get_cifar10_loaders
from .validation import validate_config, validate_optimizer_config, validate_lr_schedule_config

__all__ = [
    # Optimizers
    'SGD',
    'SGDMomentum',
    'RMSProp',
    'Adam',
    'AdamW',
    # Test Functions
    'Rosenbrock',
    'IllConditionedQuadratic',
    'SaddlePoint',
    # Models
    'SimpleMLP',
    'SimpleCNN',
    'ConvNet',
    # Data
    'get_mnist_loaders',
    'get_cifar10_loaders',
    # Validation
    'validate_config',
    'validate_optimizer_config',
    'validate_lr_schedule_config',
]
