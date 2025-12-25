"""
Optimizer Registry Pattern for eliminating hardcoded optimizer lists.

This module provides a centralized registry for all optimizers, enabling:
- Configuration-driven experiment design
- No hardcoded optimizer lists in experiment scripts
- Easy addition of new optimizers
- Consistent hyperparameter management

Usage:
    # Register an optimizer
    registry.register('MyOptimizer', MyOptimizerClass, default_lr=0.01)
    
    # Create optimizer from config
    optimizer = registry.create('Adam', model.parameters(), lr=0.001)
    
    # Run experiments from config file
    for opt_config in config['optimizers']:
        opt = registry.create(**opt_config)
"""

import torch
from torch.optim import Optimizer
from typing import Dict, Type, Any, Callable, Optional, List, Union
import logging
import json
from pathlib import Path


class OptimizerRegistry:
    """
    Central registry for all optimizers with metadata and default configurations.
    
    This eliminates "Script Sprawl" by providing a single source of truth
    for optimizer definitions and hyperparameters.
    """
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._initialize_standard_optimizers()
    
    def _initialize_standard_optimizers(self):
        """Initialize registry with standard PyTorch and custom optimizers."""
        
        # === Standard PyTorch Optimizers ===
        self.register(
            name='SGD',
            optimizer_class=torch.optim.SGD,
            default_hyperparams={'lr': 0.01},
            search_space={'lr': (1e-4, 1e-1, 'log')},
            description='Stochastic Gradient Descent'
        )
        
        self.register(
            name='SGD_Momentum',
            optimizer_class=torch.optim.SGD,
            default_hyperparams={'lr': 0.01, 'momentum': 0.9},
            search_space={
                'lr': (1e-4, 1e-1, 'log'),
                'momentum': (0.5, 0.99, 'uniform')
            },
            description='SGD with Momentum'
        )
        
        self.register(
            name='SGD_Nesterov',
            optimizer_class=torch.optim.SGD,
            default_hyperparams={'lr': 0.01, 'momentum': 0.9, 'nesterov': True},
            search_space={
                'lr': (1e-4, 1e-1, 'log'),
                'momentum': (0.5, 0.99, 'uniform')
            },
            description='SGD with Nesterov Accelerated Gradient'
        )
        
        self.register(
            name='Adam',
            optimizer_class=torch.optim.Adam,
            default_hyperparams={'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8},
            search_space={
                'lr': (1e-5, 1e-2, 'log'),
                'beta1': (0.8, 0.95, 'uniform'),
                'beta2': (0.9, 0.999, 'uniform')
            },
            description='Adam: Adaptive Moment Estimation'
        )
        
        self.register(
            name='AdamW',
            optimizer_class=torch.optim.AdamW,
            default_hyperparams={'lr': 1e-3, 'betas': (0.9, 0.999), 'weight_decay': 1e-2},
            search_space={
                'lr': (1e-5, 1e-2, 'log'),
                'weight_decay': (1e-5, 1e-1, 'log')
            },
            description='AdamW: Adam with Decoupled Weight Decay'
        )
        
        self.register(
            name='RMSprop',
            optimizer_class=torch.optim.RMSprop,
            default_hyperparams={'lr': 1e-3, 'alpha': 0.99, 'eps': 1e-8},
            search_space={
                'lr': (1e-5, 1e-2, 'log'),
                'alpha': (0.9, 0.999, 'uniform')
            },
            description='RMSprop: Root Mean Square Propagation'
        )
        
        self.register(
            name='Adagrad',
            optimizer_class=torch.optim.Adagrad,
            default_hyperparams={'lr': 1e-2},
            search_space={'lr': (1e-4, 1e-1, 'log')},
            description='Adagrad: Adaptive Gradient Algorithm'
        )
        
        # === Advanced Optimizers (if available) ===
        try:
            # Torch native optimizers for reference
            from src.core.torch_native_optimizers import (
                TorchSAM, TorchLookahead
            )
            
            self.register(
                name='SAM_SGD',
                optimizer_class=TorchSAM,
                default_hyperparams={'base_optimizer': torch.optim.SGD, 'rho': 0.05, 'lr': 0.01},
                search_space={
                    'lr': (1e-4, 1e-1, 'log'),
                    'rho': (0.01, 0.2, 'uniform')
                },
                description='SAM: Sharpness-Aware Minimization with SGD'
            )
            
            self.register(
                name='Lookahead_SGD',
                optimizer_class=lambda params, **kwargs: TorchLookahead(
                    torch.optim.SGD(params, lr=kwargs.get('lr', 0.01), momentum=0.9),
                    k=kwargs.get('k', 5),
                    alpha=kwargs.get('alpha', 0.5)
                ),
                default_hyperparams={'lr': 0.01, 'k': 5, 'alpha': 0.5},
                search_space={
                    'lr': (1e-4, 1e-1, 'log'),
                    'k': (3, 10, 'int'),
                    'alpha': (0.3, 0.8, 'uniform')
                },
                description='Lookahead: k steps forward, 1 step back (with SGD)'
            )
            
            self.register(
                name='Lookahead_Adam',
                optimizer_class=lambda params, **kwargs: TorchLookahead(
                    torch.optim.Adam(params, lr=kwargs.get('lr', 1e-3)),
                    k=kwargs.get('k', 5),
                    alpha=kwargs.get('alpha', 0.5)
                ),
                default_hyperparams={'lr': 1e-3, 'k': 5, 'alpha': 0.5},
                search_space={
                    'lr': (1e-5, 1e-2, 'log'),
                    'k': (3, 10, 'int'),
                    'alpha': (0.3, 0.8, 'uniform')
                },
                description='Lookahead with Adam'
            )
            
        except ImportError:
            logging.debug("Advanced optimizers not available")
    
    def register(self,
                 name: str,
                 optimizer_class: Union[Type[Optimizer], Callable],
                 default_hyperparams: Dict[str, Any],
                 search_space: Optional[Dict[str, tuple]] = None,
                 description: str = ""):
        """
        Register an optimizer with metadata.
        
        Args:
            name: Unique optimizer name
            optimizer_class: Optimizer class or factory function
            default_hyperparams: Default hyperparameters
            search_space: Hyperparameter search space for tuning
                Format: {'param': (min, max, 'log'|'uniform'|'int')}
            description: Human-readable description
        """
        if name in self._registry:
            logging.warning("Overwriting existing optimizer: %s", name)
        
        self._registry[name] = {
            'class': optimizer_class,
            'defaults': default_hyperparams,
            'search_space': search_space or {},
            'description': description
        }
        
        logging.debug("Registered optimizer: %s", name)
    
    def create(self, name: str, params, **override_hyperparams) -> Optimizer:
        """
        Create optimizer instance.
        
        Args:
            name: Registered optimizer name
            params: Model parameters (from model.parameters())
            **override_hyperparams: Hyperparameter overrides
            
        Returns:
            Optimizer instance
        """
        if name not in self._registry:
            raise ValueError(f"Unknown optimizer: {name}. Available: {self.list_optimizers()}")
        
        config = self._registry[name]
        
        # Merge defaults with overrides
        hyperparams = config['defaults'].copy()
        hyperparams.update(override_hyperparams)
        
        # Create optimizer
        optimizer_class = config['class']
        
        try:
            optimizer = optimizer_class(params, **hyperparams)
        except TypeError:
            # Handle callable factories
            optimizer = optimizer_class(params=params, **hyperparams)
        
        logging.debug("Created optimizer %s with hyperparams: %s", name, hyperparams)
        return optimizer
    
    def get_default_hyperparams(self, name: str) -> Dict[str, Any]:
        """Get default hyperparameters for an optimizer."""
        if name not in self._registry:
            raise ValueError(f"Unknown optimizer: {name}")
        return self._registry[name]['defaults'].copy()
    
    def get_search_space(self, name: str) -> Dict[str, tuple]:
        """Get hyperparameter search space for an optimizer."""
        if name not in self._registry:
            raise ValueError(f"Unknown optimizer: {name}")
        return self._registry[name]['search_space'].copy()
    
    def list_optimizers(self) -> List[str]:
        """List all registered optimizer names."""
        return sorted(self._registry.keys())
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """Get full information about an optimizer."""
        if name not in self._registry:
            raise ValueError(f"Unknown optimizer: {name}")
        
        config = self._registry[name].copy()
        # Don't return the class object in info
        info = {
            'name': name,
            'description': config['description'],
            'default_hyperparams': config['defaults'],
            'search_space': config['search_space']
        }
        return info
    
    def print_registry(self):
        """Print all registered optimizers."""
        print("\n" + "="*70)
        print("REGISTERED OPTIMIZERS")
        print("="*70)
        
        for name in self.list_optimizers():
            info = self.get_info(name)
            print(f"\n{name}")
            print(f"  Description: {info['description']}")
            print(f"  Defaults: {info['default_hyperparams']}")
            if info['search_space']:
                print(f"  Search space: {info['search_space']}")
        
        print("\n" + "="*70)
    
    def load_from_config(self, config_path: Union[str, Path]):
        """
        Load optimizer configurations from JSON file.
        
        Config format:
        {
            "optimizers": [
                {
                    "name": "CustomAdam",
                    "base": "Adam",
                    "hyperparams": {"lr": 0.001, "weight_decay": 1e-4}
                }
            ]
        }
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        for opt_config in config.get('optimizers', []):
            name = opt_config['name']
            base = opt_config.get('base')
            hyperparams = opt_config.get('hyperparams', {})
            
            if base:
                # Create variant of existing optimizer
                base_config = self._registry[base]
                new_defaults = base_config['defaults'].copy()
                new_defaults.update(hyperparams)
                
                self.register(
                    name=name,
                    optimizer_class=base_config['class'],
                    default_hyperparams=new_defaults,
                    search_space=base_config['search_space'],
                    description=f"Custom variant of {base}"
                )
        
        logging.info("Loaded %d optimizer configs from %s", len(config.get('optimizers', [])), config_path)
    
    def save_to_config(self, config_path: Union[str, Path], optimizer_names: Optional[List[str]] = None):
        """
        Save optimizer configurations to JSON file.
        
        Args:
            config_path: Path to save config
            optimizer_names: Specific optimizers to save (None = all)
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        optimizers_to_save = optimizer_names or self.list_optimizers()
        
        config = {
            'optimizers': []
        }
        
        for name in optimizers_to_save:
            info = self.get_info(name)
            config['optimizers'].append({
                'name': name,
                'description': info['description'],
                'default_hyperparams': info['default_hyperparams'],
                'search_space': info['search_space']
            })
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logging.info("Saved %d optimizer configs to %s", len(config['optimizers']), config_path)


# Global registry instance
registry = OptimizerRegistry()


def create_optimizer_from_config(config: Dict, model_params) -> Optimizer:
    """
    Convenience function to create optimizer from config dict.
    
    Args:
        config: Dict with 'name' and optional hyperparameter overrides
        model_params: Model parameters
        
    Returns:
        Optimizer instance
    """
    name = config['name']
    hyperparams = {k: v for k, v in config.items() if k != 'name'}
    return registry.create(name, model_params, **hyperparams)


def load_experiment_config(config_path: Union[str, Path]) -> List[Dict]:
    """
    Load experiment configuration from JSON file.
    
    Returns list of optimizer configurations for experiments.
    
    Example config:
    {
        "experiment": "CIFAR10_ResNet18",
        "optimizers": [
            {"name": "Adam", "lr": 0.001},
            {"name": "SGD_Momentum", "lr": 0.01, "momentum": 0.9}
        ]
    }
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config.get('optimizers', [])


def normalize_optimizer_name(name: str) -> str:
    """Normalize an optimizer name string to a registered canonical name.

    Rules:
      - Attempts direct match first
      - Replaces spaces and dashes with underscores
      - Case-insensitive match
      - Maps common alias patterns (e.g., 'SGDMomentum' -> 'SGD_Momentum')

    Raises:
        ValueError: If no mapping exists
    """
    if not isinstance(name, str):
        raise ValueError("Optimizer name must be a string")

    # Direct match
    if name in registry._registry:
        return name

    # Normalize separators and case
    normalized = name.replace('-', '_').replace(' ', '_')
    if normalized in registry._registry:
        return normalized

    # Lowercase matching with underscores removed/added
    low = normalized.lower()
    for canon in registry.list_optimizers():
        if canon.lower() == low or canon.lower().replace('_', '') == low.replace('_', ''):
            return canon

    # Specific common alias fixes
    alias_map = {
        'sgdmomentum': 'SGD_Momentum',
        'sgdnesterov': 'SGD_Nesterov',
        'rmsprop': 'RMSprop',
        'adamw': 'AdamW',
        'amsgrad': 'AMSGrad',
        'sam_sgd': 'SAM_SGD',
        'sam_adam': 'SAM_Adam',
        'lookahead_sgd': 'Lookahead_SGD',
        'lookahead_adam': 'Lookahead_Adam',
        'adabound': 'AdaBound',
        'radam': 'RAdam',
        'lamb': 'LAMB'
    }

    if low.replace('_', '') in alias_map:
        return alias_map[low.replace('_', '')]

    raise ValueError(f"Unknown optimizer name: '{name}'. Available: {registry.list_optimizers()}")


# Example usage function
def example_usage():
    """Demonstrate registry usage."""

    # Print all registered optimizers
    registry.print_registry()

    # Create optimizer from registry
    import torch.nn as nn
    model = nn.Linear(10, 2)

    # Method 1: Direct creation
    optimizer = registry.create('Adam', model.parameters(), lr=0.001)
    print(f"\nCreated optimizer: {optimizer}")

    # Method 2: From config dict
    config = {'name': 'SGDMomentum', 'lr': 0.01, 'momentum': 0.9}
    try:
        # Normalize name before creating (accepts 'SGDMomentum' as alias)
        config_normalized = config.copy()
        config_normalized['name'] = normalize_optimizer_name(config['name'])
        optimizer2 = create_optimizer_from_config(config_normalized, model.parameters())
        print(f"Created from config: {optimizer2}")
    except ValueError as e:
        print(f"Could not create optimizer from config: {e}")

    # Method 3: Get defaults and modify
    defaults = registry.get_default_hyperparams('AdamW')
    defaults['lr'] = 0.0005
    optimizer3 = registry.create('AdamW', model.parameters(), **defaults)
    print(f"Created with modified defaults: {optimizer3}")


if __name__ == '__main__':
    example_usage()
