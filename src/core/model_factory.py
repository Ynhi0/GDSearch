"""
Model Factory - Eliminates if/elif chains for model creation.

This module provides a clean factory pattern for creating models,
similar to the OptimizerFactory but for neural network architectures.

Features:
- Registry-based model creation
- Dataset-specific model defaults
- Easy integration with experiments
- Extensible for custom models

Example:
    >>> from src.core.model_factory import ModelFactory
    >>> model = ModelFactory.create('ResNet18', num_classes=10)
    >>> 
    >>> # From config
    >>> model_config = {'name': 'SimpleCNN', 'num_classes': 10}
    >>> model = ModelFactory.create_from_config(model_config)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type, Callable, List
import logging


class ModelFactory:
    """
    Factory for creating models with consistent interface.
    
    Eliminates scattered model instantiation code and provides
    centralized model creation with validation.
    """
    
    _REGISTRY: Dict[str, Callable[..., nn.Module]] = {}
    _DEFAULTS: Dict[str, Dict[str, Any]] = {}
    _DESCRIPTIONS: Dict[str, str] = {}
    _initialized: bool = False
    
    @classmethod
    def _initialize_registry(cls):
        """Initialize model registry with standard architectures."""
        if cls._initialized:
            return
        
        # === Standard Models ===
        try:
            from src.core.models import (
                SimpleCNN, SimpleResNet, ResNet18ForCIFAR
            )
            
            cls._REGISTRY['simplecnn'] = SimpleCNN
            cls._DEFAULTS['simplecnn'] = {'num_classes': 10, 'input_channels': 1}
            cls._DESCRIPTIONS['simplecnn'] = 'Simple CNN for MNIST (2 conv + 2 fc layers)'
            
            cls._REGISTRY['simpleresnet'] = SimpleResNet
            cls._DEFAULTS['simpleresnet'] = {'num_classes': 10}
            cls._DESCRIPTIONS['simpleresnet'] = 'Lightweight ResNet for small datasets'
            
            cls._REGISTRY['resnet18'] = ResNet18ForCIFAR
            cls._DEFAULTS['resnet18'] = {'num_classes': 10}
            cls._DESCRIPTIONS['resnet18'] = 'ResNet-18 adapted for CIFAR-10 (32x32 images)'
            
        except ImportError as e:
            logging.debug(f"Could not import standard models: {e}")
        
        # === Torchvision Models (if available) ===
        try:
            import torchvision.models as tv_models
            
            # ResNet variants
            cls._REGISTRY['resnet18_imagenet'] = lambda **kwargs: tv_models.resnet18(
                num_classes=kwargs.get('num_classes', 1000)
            )
            cls._DEFAULTS['resnet18_imagenet'] = {'num_classes': 1000}
            cls._DESCRIPTIONS['resnet18_imagenet'] = 'ResNet-18 for ImageNet (224x224)'
            
            cls._REGISTRY['resnet34'] = lambda **kwargs: tv_models.resnet34(
                num_classes=kwargs.get('num_classes', 1000)
            )
            cls._DEFAULTS['resnet34'] = {'num_classes': 1000}
            
            cls._REGISTRY['resnet50'] = lambda **kwargs: tv_models.resnet50(
                num_classes=kwargs.get('num_classes', 1000)
            )
            cls._DEFAULTS['resnet50'] = {'num_classes': 1000}
            
            # VGG variants
            cls._REGISTRY['vgg16'] = lambda **kwargs: tv_models.vgg16(
                num_classes=kwargs.get('num_classes', 1000)
            )
            cls._DEFAULTS['vgg16'] = {'num_classes': 1000}
            
            # EfficientNet
            try:
                cls._REGISTRY['efficientnet_b0'] = lambda **kwargs: tv_models.efficientnet_b0(
                    num_classes=kwargs.get('num_classes', 1000)
                )
                cls._DEFAULTS['efficientnet_b0'] = {'num_classes': 1000}
            except AttributeError:
                pass  # EfficientNet not available in this torchvision version
            
        except ImportError:
            logging.debug("Torchvision models not available")
        
        # === Medical/Segmentation Models ===
        try:
            # Check if UNet is defined in run_all_kaggle.py or separate module
            # For now, provide placeholder - user should register their U-Net
            cls._REGISTRY['unet'] = cls._create_unet_placeholder
            cls._DEFAULTS['unet'] = {
                'in_channels': 1,
                'out_channels': 1,
                'features': [64, 128, 256, 512]
            }
            cls._DESCRIPTIONS['unet'] = 'U-Net for medical image segmentation'
        except Exception:
            logging.debug("U-Net model not available")
        
        # === NLP Models (Transformers) ===
        try:
            from src.core.nlp_models import TransformerClassifier
            cls._REGISTRY['transformer'] = TransformerClassifier
            cls._DEFAULTS['transformer'] = {
                'num_classes': 2,
                'model_name': 'bert-base-uncased'
            }
            cls._DESCRIPTIONS['transformer'] = 'Transformer-based text classifier'
        except ImportError:
            logging.debug("NLP models not available")
        
        cls._initialized = True
        logging.debug(f"Model factory initialized with {len(cls._REGISTRY)} models")
    
    @staticmethod
    def _create_unet_placeholder(**kwargs):
        """
        Placeholder for U-Net creation.
        
        This is a fallback if U-Net is not properly registered.
        Users should register their actual U-Net implementation.
        """
        logging.warning(
            "Using placeholder U-Net. Register your U-Net implementation with:\n"
            "ModelFactory.register('unet', YourUNetClass)"
        )
        raise NotImplementedError(
            "U-Net not implemented. Register your U-Net model:\n"
            "from your_module import UNet\n"
            "ModelFactory.register('unet', UNet)"
        )
    
    @classmethod
    def create(
        cls,
        name: str,
        **kwargs
    ) -> nn.Module:
        """
        Create model by name.
        
        Args:
            name: Model name (case-insensitive)
            **kwargs: Model-specific arguments (e.g., num_classes, input_channels)
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is unknown
            
        Example:
            >>> model = ModelFactory.create('ResNet18', num_classes=10)
            >>> model = ModelFactory.create('SimpleCNN', num_classes=10, input_channels=1)
        """
        cls._initialize_registry()
        
        name_lower = name.lower().replace('_', '').replace('-', '')
        
        if name_lower not in cls._REGISTRY:
            available = ', '.join(sorted(cls._REGISTRY.keys()))
            raise ValueError(
                f"Unknown model: {name}\n"
                f"Available models: {available}\n"
                f"Use ModelFactory.register() to add custom models"
            )
        
        model_fn = cls._REGISTRY[name_lower]
        
        # Apply defaults
        defaults = cls._DEFAULTS.get(name_lower, {})
        final_kwargs = {**defaults, **kwargs}
        
        # Create model
        try:
            model = model_fn(**final_kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to create model {name} with arguments {final_kwargs}: {e}\n"
                f"Check that all arguments are valid for this model."
            ) from e
        
        logging.debug(f"Created model: {name}")
        return model
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        """
        Create model from configuration dictionary.
        
        Args:
            config: Configuration dict with 'name' and model parameters
            
        Returns:
            Model instance
            
        Example:
            >>> config = {'name': 'ResNet18', 'num_classes': 10}
            >>> model = ModelFactory.create_from_config(config)
        """
        if 'name' not in config:
            raise ValueError(
                "Config must contain 'name' field specifying model name.\n"
                f"Got config: {config}"
            )
        
        name = config['name']
        kwargs = {k: v for k, v in config.items() if k != 'name'}
        
        return cls.create(name, **kwargs)
    
    @classmethod
    def register(
        cls,
        name: str,
        model_fn: Callable[..., nn.Module],
        default_params: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> None:
        """
        Register custom model.
        
        Args:
            name: Model name (case-insensitive)
            model_fn: Function/class that creates model (must accept **kwargs)
            default_params: Optional default parameters
            description: Model description
            
        Example:
            >>> class MyModel(nn.Module):
            ...     def __init__(self, num_classes=10):
            ...         super().__init__()
            ...         self.fc = nn.Linear(784, num_classes)
            ...
            >>> ModelFactory.register(
            ...     'MyModel',
            ...     MyModel,
            ...     default_params={'num_classes': 10},
            ...     description='Custom linear model'
            ... )
        """
        cls._initialize_registry()
        
        name_lower = name.lower()
        
        if name_lower in cls._REGISTRY:
            logging.warning(f"Model '{name}' already registered. Overwriting.")
        
        cls._REGISTRY[name_lower] = model_fn
        
        if default_params is not None:
            cls._DEFAULTS[name_lower] = default_params
        
        if description:
            cls._DESCRIPTIONS[name_lower] = description
        
        logging.info(f"Registered model: {name}")
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        Get list of all registered model names.
        
        Returns:
            Sorted list of model names
        """
        cls._initialize_registry()
        return sorted(cls._REGISTRY.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if model is registered.
        
        Args:
            name: Model name
            
        Returns:
            True if registered, False otherwise
        """
        cls._initialize_registry()
        return name.lower() in cls._REGISTRY
    
    @classmethod
    def get_description(cls, name: str) -> str:
        """
        Get model description.
        
        Args:
            name: Model name
            
        Returns:
            Model description string
        """
        cls._initialize_registry()
        name_lower = name.lower()
        return cls._DESCRIPTIONS.get(name_lower, "No description available")
    
    @classmethod
    def get_defaults(cls, name: str) -> Dict[str, Any]:
        """
        Get default parameters for model.
        
        Args:
            name: Model name
            
        Returns:
            Dictionary of default parameters
        """
        cls._initialize_registry()
        name_lower = name.lower()
        return cls._DEFAULTS.get(name_lower, {}).copy()


def create_model_for_dataset(
    model_name: str,
    dataset_name: str,
    **kwargs
) -> nn.Module:
    """
    Create model configured for specific dataset.
    
    Automatically sets appropriate num_classes, input_channels, etc.
    based on dataset.
    
    Args:
        model_name: Name of model architecture
        dataset_name: Dataset name ('mnist', 'cifar10', etc.)
        **kwargs: Additional model parameters (override defaults)
        
    Returns:
        Configured model instance
        
    Example:
        >>> model = create_model_for_dataset('SimpleCNN', 'mnist')
        >>> # Automatically sets num_classes=10, input_channels=1
    """
    from src.utils.constants import (
        MNIST_NUM_CLASSES, MNIST_IMAGE_SIZE,
        CIFAR10_NUM_CLASSES, CIFAR10_IMAGE_SIZE
    )
    
    dataset_config = {
        'mnist': {
            'num_classes': MNIST_NUM_CLASSES,
            'input_channels': 1,
            'input_size': MNIST_IMAGE_SIZE,
        },
        'cifar10': {
            'num_classes': CIFAR10_NUM_CLASSES,
            'input_channels': 3,
            'input_size': CIFAR10_IMAGE_SIZE,
        },
        'cifar-10': {
            'num_classes': CIFAR10_NUM_CLASSES,
            'input_channels': 3,
            'input_size': CIFAR10_IMAGE_SIZE,
        },
    }
    
    dataset_key = dataset_name.lower()
    if dataset_key not in dataset_config:
        logging.warning(
            f"Unknown dataset '{dataset_name}', using provided kwargs only"
        )
        dataset_defaults = {}
    else:
        dataset_defaults = dataset_config[dataset_key]
    
    # Merge dataset defaults with user kwargs (kwargs take precedence)
    final_kwargs = {**dataset_defaults, **kwargs}
    
    return ModelFactory.create(model_name, **final_kwargs)


# Convenience function
def create_model(name: str, **kwargs) -> nn.Module:
    """
    Convenience function to create model.
    
    Equivalent to ModelFactory.create() but shorter.
    
    Args:
        name: Model name
        **kwargs: Model parameters
        
    Returns:
        Model instance
    """
    return ModelFactory.create(name, **kwargs)
