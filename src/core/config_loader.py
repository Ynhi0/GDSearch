"""
Centralized configuration loading and validation.

This module eliminates configuration parsing duplication across scripts
and provides a single source of truth for configuration handling.

Features:
- JSON config file loading with validation
- Deep dictionary merging for config overrides
- Default value application
- Type checking and validation
- Config schema validation

Example:
    >>> from src.core.config_loader import ConfigLoader
    >>> config = ConfigLoader.load_experiment_config('configs/nn_tuning.json')
    >>> config = ConfigLoader.apply_defaults(config, default_config)
    >>> ConfigLoader.validate_config(config, schema)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from copy import deepcopy


# Dataset-Model Compatibility Matrix
# Ensures models are only used with compatible datasets
DATASET_MODEL_COMPATIBILITY = {
    "MNIST": ["SimpleMLP", "SimpleCNN"],
    "FashionMNIST": ["SimpleMLP", "SimpleCNN"],
    "CIFAR10": ["SimpleCNN", "ConvNet", "ResNet18"],
    "CIFAR100": ["ConvNet", "ResNet18"],
    "IMDB": ["SimpleRNN", "SimpleLSTM", "BiLSTM", "TextCNN"],
    "PathMNIST": ["SimpleCNN", "ConvNet"]
}


class ConfigLoader:
    """Centralized configuration loading and management."""
    
    # Default configurations for common experiment types
    DEFAULT_MNIST_CONFIG = {
        'batch_size': 128,
        'epochs': 50,
        'patience': 10,
        'val_split': 0.15,
        'device': 'cuda',
        'num_workers': 2,
        'pin_memory': True,
    }
    
    DEFAULT_CIFAR_CONFIG = {
        'batch_size': 128,
        'epochs': 100,
        'patience': 15,
        'val_split': 0.15,
        'device': 'cuda',
        'num_workers': 4,
        'pin_memory': True,
    }
    
    DEFAULT_NLP_CONFIG = {
        'batch_size': 32,
        'epochs': 10,
        'max_length': 128,
        'device': 'cuda',
        'num_workers': 2,
    }
    
    DEFAULT_MEDICAL_CONFIG = {
        'batch_size': 16,
        'epochs': 50,
        'image_size': 128,
        'device': 'cuda',
        'num_workers': 2,
    }
    
    @staticmethod
    def load_experiment_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate experiment configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            ValueError: If required fields are missing
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create the config file or check the path."
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in config file {config_path}: {e.msg}",
                e.doc,
                e.pos
            ) from e
        
        # Log successful load
        logging.info(f"Loaded configuration from {config_path}")
        
        return config
    
    @staticmethod
    def merge_configs(
        base: Dict[str, Any],
        override: Dict[str, Any],
        *,
        deep: bool = True
    ) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Override values take precedence. Nested dictionaries are merged recursively
        if deep=True, otherwise override replaces base entirely.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            deep: If True, recursively merge nested dicts; if False, replace entirely
            
        Returns:
            Merged configuration dictionary
            
        Example:
            >>> base = {'a': 1, 'b': {'x': 10, 'y': 20}}
            >>> override = {'b': {'x': 15, 'z': 30}, 'c': 3}
            >>> merged = ConfigLoader.merge_configs(base, override)
            >>> # Result: {'a': 1, 'b': {'x': 15, 'y': 20, 'z': 30}, 'c': 3}
        """
        # Deep copy to avoid mutating inputs
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict) and deep:
                # Recursively merge nested dictionaries
                result[key] = ConfigLoader.merge_configs(result[key], value, deep=deep)
            else:
                # Override value
                result[key] = deepcopy(value)
        
        return result
    
    @staticmethod
    def apply_defaults(
        config: Dict[str, Any],
        defaults: Dict[str, Any],
        *,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Apply default values for missing keys in configuration.
        
        Args:
            config: Configuration dictionary (may have missing keys)
            defaults: Default values to apply
            overwrite: If True, overwrite existing values; if False, only fill missing
            
        Returns:
            Configuration with defaults applied
            
        Example:
            >>> config = {'batch_size': 64}
            >>> defaults = {'batch_size': 128, 'epochs': 50, 'lr': 0.001}
            >>> result = ConfigLoader.apply_defaults(config, defaults)
            >>> # Result: {'batch_size': 64, 'epochs': 50, 'lr': 0.001}
        """
        result = deepcopy(config)
        
        for key, default_value in defaults.items():
            if key not in result or overwrite:
                result[key] = deepcopy(default_value)
            elif isinstance(result[key], dict) and isinstance(default_value, dict):
                # Recursively apply defaults to nested dicts
                result[key] = ConfigLoader.apply_defaults(
                    result[key],
                    default_value,
                    overwrite=overwrite
                )
        
        return result
    
    @staticmethod
    def validate_required_fields(
        config: Dict[str, Any],
        required_fields: List[str]
    ) -> None:
        """
        Validate that required fields are present in configuration.
        
        Args:
            config: Configuration dictionary to validate
            required_fields: List of required field names (supports nested keys with '.')
            
        Raises:
            ValueError: If any required field is missing
            
        Example:
            >>> config = {'model': {'type': 'resnet'}, 'optimizer': 'adam'}
            >>> ConfigLoader.validate_required_fields(
            ...     config,
            ...     ['model.type', 'optimizer', 'learning_rate']
            ... )
            >>> # Raises: ValueError - 'learning_rate' is missing
        """
        missing_fields = []
        
        for field in required_fields:
            # Support nested field access with dot notation
            keys = field.split('.')
            current = config
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    missing_fields.append(field)
                    break
        
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {missing_fields}\n"
                f"Please add these fields to your config file."
            )
    
    @staticmethod
    def validate_types(
        config: Dict[str, Any],
        type_spec: Dict[str, type]
    ) -> None:
        """
        Validate types of configuration values.
        
        Args:
            config: Configuration dictionary to validate
            type_spec: Dictionary mapping field names to expected types
            
        Raises:
            TypeError: If any field has incorrect type
            
        Example:
            >>> config = {'batch_size': 128, 'lr': 0.001}
            >>> ConfigLoader.validate_types(config, {'batch_size': int, 'lr': float})
        """
        type_errors = []
        
        for field, expected_type in type_spec.items():
            keys = field.split('.')
            current = config
            
            for key in keys[:-1]:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    break  # Field doesn't exist, skip type check
            else:
                # Check final key
                final_key = keys[-1]
                if final_key in current:
                    value = current[final_key]
                    if not isinstance(value, expected_type):
                        type_errors.append(
                            f"{field}: expected {expected_type.__name__}, "
                            f"got {type(value).__name__} ({value})"
                        )
        
        if type_errors:
            raise TypeError(
                f"Configuration type errors:\n" +
                "\n".join(f"  - {error}" for error in type_errors)
            )
    
    @staticmethod
    def get_dataset_defaults(dataset_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific dataset.
        
        Args:
            dataset_name: Name of dataset ('mnist', 'cifar10', 'nlp', 'medical')
            
        Returns:
            Default configuration dictionary for the dataset
            
        Raises:
            ValueError: If dataset name is unknown
        """
        dataset_defaults = {
            'mnist': ConfigLoader.DEFAULT_MNIST_CONFIG,
            'cifar10': ConfigLoader.DEFAULT_CIFAR_CONFIG,
            'cifar-10': ConfigLoader.DEFAULT_CIFAR_CONFIG,
            'nlp': ConfigLoader.DEFAULT_NLP_CONFIG,
            'medical': ConfigLoader.DEFAULT_MEDICAL_CONFIG,
            'segmentation': ConfigLoader.DEFAULT_MEDICAL_CONFIG,
        }
        
        dataset_key = dataset_name.lower()
        if dataset_key not in dataset_defaults:
            available = ', '.join(sorted(set(dataset_defaults.keys())))
            raise ValueError(
                f"Unknown dataset: {dataset_name}\n"
                f"Available datasets: {available}"
            )
        
        return deepcopy(dataset_defaults[dataset_key])
    
    @staticmethod
    def save_config(
        config: Dict[str, Any],
        output_path: Union[str, Path],
        *,
        indent: int = 2
    ) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary to save
            output_path: Path to output JSON file
            indent: JSON indentation level (default: 2)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=indent, sort_keys=True)
        
        logging.info(f"Saved configuration to {output_path}")
    
    @staticmethod
    def create_experiment_config(
        dataset: str,
        optimizers: List[str],
        learning_rates: Optional[Dict[str, float]] = None,
        seeds: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a complete experiment configuration.
        
        Args:
            dataset: Dataset name ('mnist', 'cifar10', etc.)
            optimizers: List of optimizer names to test
            learning_rates: Optional dict mapping optimizer names to learning rates
            seeds: Optional list of random seeds
            **kwargs: Additional config parameters
            
        Returns:
            Complete experiment configuration dictionary
            
        Example:
            >>> config = ConfigLoader.create_experiment_config(
            ...     dataset='mnist',
            ...     optimizers=['SGD', 'Adam', 'AdamW'],
            ...     learning_rates={'SGD': 0.1, 'Adam': 0.001, 'AdamW': 0.001},
            ...     seeds=[42, 123, 456],
            ...     epochs=50
            ... )
        """
        # Start with dataset defaults
        config = ConfigLoader.get_dataset_defaults(dataset)
        
        # Add experiment-specific fields
        config['dataset'] = dataset
        config['optimizers'] = optimizers
        
        if learning_rates is not None:
            config['learning_rates'] = learning_rates
        
        if seeds is not None:
            config['seeds'] = seeds
        else:
            config['seeds'] = [42]  # Default single seed
        
        # Merge additional kwargs
        config.update(kwargs)
        
        return config


class ConfigValidator:
    """Validates experiment configurations against schemas."""
    
    @staticmethod
    def validate_optimizer_config(config: Dict[str, Any]) -> None:
        """
        Validate optimizer configuration.
        
        Args:
            config: Optimizer configuration dict
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['optimizers']
        ConfigLoader.validate_required_fields(config, required_fields)
        
        # Validate optimizer names (use factory instead of registry)
        try:
            from src.core.optimizer_factory import OptimizerFactory
            
            for opt_name in config['optimizers']:
                if not OptimizerFactory.is_registered(opt_name):
                    available = ', '.join(sorted(OptimizerFactory.list_optimizers()))
                    raise ValueError(
                        f"Unknown optimizer: {opt_name}\n"
                        f"Available optimizers: {available}\n"
                        f"Use OptimizerFactory.register() to add custom optimizers"
                    )
        except ImportError:
            # Optimizer factory not available - skip validation
            logging.debug("OptimizerFactory not available, skipping optimizer validation")
    
    @staticmethod
    def validate_experiment_config(config: Dict[str, Any]) -> None:
        """
        Validate complete experiment configuration.
        
        Args:
            config: Full experiment configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['dataset', 'optimizers', 'epochs']
        ConfigLoader.validate_required_fields(config, required_fields)
        
        # Type validation
        type_spec = {
            'epochs': int,
            'batch_size': int,
            'optimizers': list,
        }
        ConfigLoader.validate_types(config, type_spec)
        
        # Validate optimizer config
        ConfigValidator.validate_optimizer_config(config)
        
        # Validate epochs > 0
        if config['epochs'] <= 0:
            raise ValueError(f"epochs must be positive, got {config['epochs']}")
        
        # Validate batch_size > 0 if present
        if 'batch_size' in config and config['batch_size'] <= 0:
            raise ValueError(f"batch_size must be positive, got {config['batch_size']}")
        
        logging.info("✓ Configuration validation passed")


def load_and_validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load and validate configuration in one step.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config = ConfigLoader.load_experiment_config(config_path)
    ConfigValidator.validate_experiment_config(config)
    validate_config_compatibility(config)  # H2: Added compatibility check
    return config


def validate_config_compatibility(config: Dict[str, Any]) -> None:
    """
    Validate dataset-model compatibility (standalone function).
    
    Ensures that the model architecture is compatible with the dataset format.
    This prevents common errors like using text models on image data or vice versa.
    
    Args:
        config: Experiment configuration dictionary
        
    Raises:
        ValueError: If model is incompatible with dataset
        
    Example:
        >>> config = {"dataset": "CIFAR10", "model": "SimpleLSTM"}
        >>> validate_config_compatibility(config)
        ValueError: Invalid model 'SimpleLSTM' for dataset 'CIFAR10'...
    """
    dataset = config.get("dataset")
    model = config.get("model")
    
    if dataset and model:
        valid_models = DATASET_MODEL_COMPATIBILITY.get(dataset, [])
        if valid_models and model not in valid_models:
            raise ValueError(
                f"Invalid model '{model}' for dataset '{dataset}'. "
                f"Valid models for {dataset}: {', '.join(valid_models)}\n"
                f"REASON: {model} architecture is incompatible with {dataset} data format. "
                f"Check input dimensions and model architecture requirements.\n"
                f"Example: Text models (LSTM, RNN) require sequential data; "
                f"CNNs require image data with spatial structure."
            )
        elif dataset and not valid_models:
            logging.warning(
                f"Dataset '{dataset}' not in compatibility matrix. "
                f"Skipping model compatibility check. "
                f"Add to DATASET_MODEL_COMPATIBILITY in config_loader.py if needed."
            )
