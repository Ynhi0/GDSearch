"""
Core implementations: optimizers, test functions, models, data utilities, LR schedulers, Optuna tuning, advanced training utilities.
"""

from .optimizers import SGD, SGDMomentum, RMSProp, Adam
from .test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint
from .validation import validate_config, validate_learning_rate, validate_epochs, validate_batch_size
from .lr_schedulers import (
    LRScheduler, ConstantLR, StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearWarmupScheduler,
    PolynomialLR, OneCycleLR, get_scheduler
)

# Optional imports - gracefully handle missing dependencies
try:
    from .optuna_tuner import (
        OptunaHyperparameterTuner, suggest_optimizer_params,
        suggest_lr_scheduler_params, suggest_model_params, suggest_training_params
    )
except ImportError:
    OptunaHyperparameterTuner = None
    suggest_optimizer_params = None
    suggest_lr_scheduler_params = None
    suggest_model_params = None
    suggest_training_params = None

try:
    from .training_utils import (
        LabelSmoothingCrossEntropy, ModelEMA, AMPWrapper,
        get_loss_function, create_amp_wrapper, create_model_ema
    )
except ImportError:
    LabelSmoothingCrossEntropy = None
    ModelEMA = None
    AMPWrapper = None
    get_loss_function = None
    create_amp_wrapper = None
    create_model_ema = None

# Optional imports that may pull torch/torchvision and fail on some systems.
# Keep core test-function utilities usable even if torch DLLs fail to load.
try:
    from .models import SimpleMLP, SimpleCNN, ConvNet
except Exception:
    SimpleMLP = None
    SimpleCNN = None
    ConvNet = None

try:
    from .data_utils import get_mnist_loaders, get_cifar10_loaders
except Exception:
    get_mnist_loaders = None
    get_cifar10_loaders = None

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
