"""
Optimizer Factory - Eliminates if/elif chains for optimizer creation.

This module provides a clean factory pattern for creating optimizers,
replacing the 15+ if/elif chains scattered throughout the codebase.

Features:
- Registry-based optimizer creation
- Automatic hyperparameter application
- Type-safe optimizer instantiation
- Easy extension for custom optimizers

Example:
    >>> from src.core.optimizer_factory import OptimizerFactory
    >>> factory = OptimizerFactory()
    >>> optimizer = factory.create('Adam', model.parameters(), lr=0.001)
    >>> 
    >>> # With config dict
    >>> opt_config = {'name': 'SGD', 'lr': 0.1, 'momentum': 0.9}
    >>> optimizer = factory.create_from_config(model.parameters(), opt_config)
"""

import torch
from torch.optim import Optimizer
from typing import Dict, Any, Optional, Iterable, Type
import logging


class OptimizerFactory:
    """
    Factory for creating optimizers with consistent interface.
    
    This eliminates the need for long if/elif chains when creating optimizers
    and provides a single source of truth for optimizer instantiation.
    """
    
    # Registry of available optimizers (name -> class)
    _REGISTRY: Dict[str, Type[Optimizer]] = {}
    
    # Default hyperparameters for each optimizer
    _DEFAULTS: Dict[str, Dict[str, Any]] = {}
    
    # Flag to track if registry is initialized
    _initialized: bool = False
    
    @classmethod
    def _initialize_registry(cls):
        """Initialize the optimizer registry with standard optimizers."""
        if cls._initialized:
            return
        
        # === PyTorch Built-in Optimizers ===
        cls._REGISTRY.update({
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'adamax': torch.optim.Adamax,
            'rmsprop': torch.optim.RMSprop,
            'adagrad': torch.optim.Adagrad,
            'adadelta': torch.optim.Adadelta,
            'rprop': torch.optim.Rprop,
            'asgd': torch.optim.ASGD,
            'lbfgs': torch.optim.LBFGS,
        })
        
        # === Default Hyperparameters ===
        cls._DEFAULTS.update({
            'sgd': {'lr': 0.1},
            'sgd_momentum': {'lr': 0.1, 'momentum': 0.9},
            'sgd_nesterov': {'lr': 0.1, 'momentum': 0.9, 'nesterov': True},
            'adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8},
            'adamw': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01},
            'adamax': {'lr': 0.002, 'betas': (0.9, 0.999)},
            'rmsprop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-8},
            'adagrad': {'lr': 0.01},
            'adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6},
            'amsgrad': {'lr': 0.001, 'betas': (0.9, 0.999), 'amsgrad': True},
        })
        
        # === Custom Optimizers (if available) ===
        try:
            from src.core.optimizers import SAM, Lookahead
            cls._REGISTRY['sam'] = SAM
            cls._REGISTRY['lookahead'] = Lookahead
            cls._DEFAULTS['sam'] = {'lr': 0.1, 'rho': 0.05}
            cls._DEFAULTS['lookahead'] = {'lr': 0.001, 'k': 5, 'alpha': 0.5}
        except ImportError:
            logging.debug("Custom optimizers (SAM, Lookahead) not available")
        
        # === PyTorch Wrappers ===
        try:
            from src.core.pytorch_optimizers import (
                AdaBoundOptimizer, RAdamOptimizer, LAMBOptimizer
            )
            cls._REGISTRY['adabound'] = AdaBoundOptimizer
            cls._REGISTRY['radam'] = RAdamOptimizer
            cls._REGISTRY['lamb'] = LAMBOptimizer
            cls._DEFAULTS['adabound'] = {'lr': 0.001, 'final_lr': 0.1}
            cls._DEFAULTS['radam'] = {'lr': 0.001}
            cls._DEFAULTS['lamb'] = {'lr': 0.001, 'weight_decay': 0.01}
        except ImportError:
            logging.debug("PyTorch optimizer wrappers not available")
        
        cls._initialized = True
        logging.debug(f"Optimizer factory initialized with {len(cls._REGISTRY)} optimizers")
    
    @classmethod
    def create(
        cls,
        name: str,
        params: Iterable,
        *,
        lr: Optional[float] = None,
        **kwargs
    ) -> Optimizer:
        """
        Create optimizer by name with parameters.
        
        Args:
            name: Optimizer name (case-insensitive)
            params: Model parameters to optimize
            lr: Learning rate (uses default if None)
            **kwargs: Additional optimizer-specific hyperparameters
            
        Returns:
            Configured optimizer instance
            
        Raises:
            ValueError: If optimizer name is unknown
            
        Example:
            >>> optimizer = OptimizerFactory.create(
            ...     'Adam',
            ...     model.parameters(),
            ...     lr=0.001,
            ...     betas=(0.9, 0.999)
            ... )
        """
        cls._initialize_registry()
        
        name_lower = name.lower().replace('_', '').replace('-', '')
        
        # Handle special cases (SGD with momentum/nesterov)
        if name_lower == 'sgdmomentum':
            name_lower = 'sgd'
            if 'momentum' not in kwargs:
                kwargs['momentum'] = 0.9
        elif name_lower == 'sgdnesterov':
            name_lower = 'sgd'
            if 'momentum' not in kwargs:
                kwargs['momentum'] = 0.9
            kwargs['nesterov'] = True
        elif name_lower == 'amsgrad':
            name_lower = 'adam'
            kwargs['amsgrad'] = True
        
        # Lookup optimizer class
        if name_lower not in cls._REGISTRY:
            available = ', '.join(sorted(cls._REGISTRY.keys()))
            raise ValueError(
                f"Unknown optimizer: {name}\n"
                f"Available optimizers: {available}\n"
                f"Use OptimizerFactory.register() to add custom optimizers"
            )
        
        optimizer_class = cls._REGISTRY[name_lower]
        
        # Apply defaults if not provided
        defaults = cls._DEFAULTS.get(name_lower, {})
        if lr is None and 'lr' in defaults:
            lr = defaults['lr']
        
        # Merge defaults with provided kwargs
        final_kwargs = {**defaults, **kwargs}
        if lr is not None:
            final_kwargs['lr'] = lr
        
        # Validate required parameters
        if 'lr' not in final_kwargs:
            raise ValueError(
                f"Learning rate (lr) must be specified for {name}. "
                f"Either provide lr argument or set default in factory."
            )
        
        # Create optimizer
        try:
            optimizer = optimizer_class(params, **final_kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to create optimizer {name} with parameters {final_kwargs}: {e}\n"
                f"Check that all hyperparameters are valid for this optimizer."
            ) from e
        
        logging.debug(f"Created optimizer: {name} with lr={final_kwargs.get('lr')}")
        return optimizer
    
    @classmethod
    def create_from_config(
        cls,
        params: Iterable,
        config: Dict[str, Any]
    ) -> Optimizer:
        """
        Create optimizer from configuration dictionary.
        
        Args:
            params: Model parameters to optimize
            config: Configuration dict with 'name' and hyperparameters
            
        Returns:
            Configured optimizer instance
            
        Example:
            >>> config = {
            ...     'name': 'SGD',
            ...     'lr': 0.1,
            ...     'momentum': 0.9,
            ...     'weight_decay': 1e-4
            ... }
            >>> optimizer = OptimizerFactory.create_from_config(
            ...     model.parameters(),
            ...     config
            ... )
        """
        if 'name' not in config:
            raise ValueError(
                "Config must contain 'name' field specifying optimizer name.\n"
                f"Got config: {config}"
            )
        
        name = config['name']
        # Extract all kwargs except 'name'
        kwargs = {k: v for k, v in config.items() if k != 'name'}
        
        return cls.create(name, params, **kwargs)
    
    @classmethod
    def register(
        cls,
        name: str,
        optimizer_class: Type[Optimizer],
        default_hyperparams: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register custom optimizer.
        
        Args:
            name: Optimizer name (case-insensitive)
            optimizer_class: Optimizer class (must inherit from torch.optim.Optimizer)
            default_hyperparams: Optional default hyperparameters
            
        Example:
            >>> class MyOptimizer(torch.optim.Optimizer):
            ...     def __init__(self, params, lr=0.01):
            ...         super().__init__(params, {'lr': lr})
            ...
            >>> OptimizerFactory.register(
            ...     'MyOptimizer',
            ...     MyOptimizer,
            ...     default_hyperparams={'lr': 0.01}
            ... )
        """
        cls._initialize_registry()
        
        name_lower = name.lower()
        
        if name_lower in cls._REGISTRY:
            logging.warning(
                f"Optimizer '{name}' is already registered. "
                f"Overwriting with new implementation."
            )
        
        cls._REGISTRY[name_lower] = optimizer_class
        
        if default_hyperparams is not None:
            cls._DEFAULTS[name_lower] = default_hyperparams
        
        logging.info(f"Registered optimizer: {name}")
    
    @classmethod
    def list_optimizers(cls) -> list[str]:
        """
        Get list of all registered optimizer names.
        
        Returns:
            Sorted list of optimizer names
        """
        cls._initialize_registry()
        return sorted(cls._REGISTRY.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if optimizer is registered.
        
        Args:
            name: Optimizer name
            
        Returns:
            True if registered, False otherwise
        """
        cls._initialize_registry()
        return name.lower() in cls._REGISTRY
    
    @classmethod
    def get_default_hyperparams(cls, name: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for optimizer.
        
        Args:
            name: Optimizer name
            
        Returns:
            Dictionary of default hyperparameters
            
        Raises:
            ValueError: If optimizer not registered
        """
        cls._initialize_registry()
        
        name_lower = name.lower()
        if name_lower not in cls._DEFAULTS:
            if name_lower in cls._REGISTRY:
                return {}  # Registered but no defaults
            else:
                raise ValueError(f"Unknown optimizer: {name}")
        
        return cls._DEFAULTS[name_lower].copy()


# Convenience instance
_factory_instance = OptimizerFactory()


def create_optimizer(
    name: str,
    params: Iterable,
    *,
    lr: Optional[float] = None,
    **kwargs
) -> Optimizer:
    """
    Convenience function to create optimizer.
    
    This is equivalent to OptimizerFactory.create() but shorter for common use.
    
    Args:
        name: Optimizer name
        params: Model parameters
        lr: Learning rate
        **kwargs: Additional hyperparameters
        
    Returns:
        Configured optimizer
        
    Example:
        >>> optimizer = create_optimizer('Adam', model.parameters(), lr=0.001)
    """
    return OptimizerFactory.create(name, params, lr=lr, **kwargs)


def get_optimizer_for_experiment(
    optimizer_name: str,
    model_params: Iterable,
    learning_rate: Optional[float] = None,
    hyperparams: Optional[Dict[str, Any]] = None
) -> Optimizer:
    """
    Create optimizer with experiment-specific hyperparameters.
    
    This function provides a high-level interface for experiment scripts,
    handling common patterns like per-optimizer default learning rates.
    
    Args:
        optimizer_name: Name of optimizer
        model_params: Model parameters to optimize
        learning_rate: Learning rate (uses optimizer-specific default if None)
        hyperparams: Additional hyperparameters
        
    Returns:
        Configured optimizer instance
    """
    hyperparams = hyperparams or {}
    
    # Apply learning rate
    if learning_rate is not None:
        hyperparams['lr'] = learning_rate
    
    return OptimizerFactory.create(optimizer_name, model_params, **hyperparams)
