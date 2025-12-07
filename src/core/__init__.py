"""
Core implementations: optimizers, test functions, models, data utilities, LR schedulers, Optuna tuning, advanced training utilities.
"""

from .optimizers import SGD, SGDMomentum, RMSProp, Adam
from .test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint
from .models import SimpleMLP, SimpleCNN, ConvNet
from .data_utils import get_mnist_loaders, get_cifar10_loaders
from .validation import validate_config, validate_learning_rate, validate_epochs, validate_batch_size
from .lr_schedulers import (
    LRScheduler, ConstantLR, StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearWarmupScheduler,
    PolynomialLR, OneCycleLR, get_scheduler
)
from .optuna_tuner import (
    OptunaHyperparameterTuner, suggest_optimizer_params,
    suggest_lr_scheduler_params, suggest_model_params, suggest_training_params
)
from .training_utils import (
    LabelSmoothingCrossEntropy, ModelEMA, AMPWrapper,
    get_loss_function, create_amp_wrapper, create_model_ema
)

__all__ = [
    # Optimizers
    'SGD',
    'SGDMomentum',
    'RMSProp',
    'Adam',
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
    'validate_learning_rate',
    'validate_epochs',
    'validate_batch_size',
    # LR Schedulers
    'LRScheduler',
    'ConstantLR',
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts',
    'LinearWarmupScheduler',
    'PolynomialLR',
    'OneCycleLR',
    'get_scheduler',
    # Optuna Tuning
    'OptunaHyperparameterTuner',
    'suggest_optimizer_params',
    'suggest_lr_scheduler_params',
    'suggest_model_params',
    'suggest_training_params',
    # Advanced Training Utilities
    'LabelSmoothingCrossEntropy',
    'ModelEMA',
    'AMPWrapper',
    'get_loss_function',
    'create_amp_wrapper',
    'create_model_ema',
]
