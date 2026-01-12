"""
Configuration loader utility for GDSearch experiments.
Loads JSON configuration files and provides type-safe access to hyperparameters.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class ConfigLoader:
    """
    Central configuration loader for all GDSearch experiments.
    Loads and validates JSON config files with schema checking.
    """

    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config JSON files
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise ConfigurationError(f"Config directory not found: {self.config_dir}")

        self._cache: Dict[str, Dict[str, Any]] = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_name: Name of config file (without .json extension)

        Returns:
            Dictionary containing configuration

        Raises:
            ConfigurationError: If file not found or invalid JSON
        """
        # Check cache first
        if config_name in self._cache:
            return self._cache[config_name]

        # Construct file path
        config_path = self.config_dir / f"{config_name}.json"
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")

        # Load and parse JSON
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {config_path}: {e}")
        except IOError as e:
            raise ConfigurationError(f"Failed to read {config_path}: {e}")

        # Cache and return
        self._cache[config_name] = config
        logging.debug("Loaded config from %s", config_path)
        return config

    def get_optimizer_config(self, config_name: str, experiment_key: str,
                           optimizer_name: str) -> Dict[str, Any]:
        """
        Get optimizer configuration for specific experiment.

        Args:
            config_name: Name of config file (e.g., 'benchmark_hyperparameters')
            experiment_key: Experiment identifier (e.g., '2d_optimization', 'resnet_cifar10')
            optimizer_name: Optimizer name (e.g., 'Adam', 'SGD')

        Returns:
            Dictionary with optimizer hyperparameters

        Raises:
            ConfigurationError: If config path invalid
        """
        config = self.load(config_name)

        # Navigate nested structure
        try:
            experiment_configs = config.get('experiment_configs', config)
            experiment = experiment_configs.get(experiment_key)
            if experiment is None:
                raise ConfigurationError(
                    f"Experiment '{experiment_key}' not found in {config_name}"
                )

            optimizers = experiment.get('optimizers', {})
            opt_config = optimizers.get(optimizer_name)
            if opt_config is None:
                raise ConfigurationError(
                    f"Optimizer '{optimizer_name}' not found in {experiment_key}"
                )

            return opt_config.copy()  # Return copy to prevent accidental modification

        except (KeyError, AttributeError, TypeError) as e:
            raise ConfigurationError(
                f"Invalid config structure in {config_name}: {e}"
            )

    def get_experiment_config(self, config_name: str, experiment_key: str) -> Dict[str, Any]:
        """
        Get full experiment configuration.

        Args:
            config_name: Name of config file
            experiment_key: Experiment identifier

        Returns:
            Dictionary with all experiment settings
        """
        config = self.load(config_name)

        try:
            experiment_configs = config.get('experiment_configs', config)
            experiment = experiment_configs.get(experiment_key)
            if experiment is None:
                raise ConfigurationError(
                    f"Experiment '{experiment_key}' not found in {config_name}"
                )

            return experiment.copy()

        except (KeyError, AttributeError, TypeError) as e:
            raise ConfigurationError(
                f"Invalid config structure in {config_name}: {e}"
            )

    def list_optimizers(self, config_name: str, experiment_key: str) -> list:
        """
        List all optimizers configured for an experiment.

        Args:
            config_name: Name of config file
            experiment_key: Experiment identifier

        Returns:
            List of optimizer names
        """
        experiment = self.get_experiment_config(config_name, experiment_key)
        return list(experiment.get('optimizers', {}).keys())


# Global singleton instance
_default_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Union[str, Path] = "configs") -> ConfigLoader:
    """
    Get or create global config loader instance.

    Args:
        config_dir: Directory containing config files

    Returns:
        ConfigLoader instance
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader(config_dir)
    return _default_loader


def load_optimizer_config(config_name: str, experiment_key: str,
                         optimizer_name: str) -> Dict[str, Any]:
    """
    Convenience function to load optimizer config using default loader.

    Args:
        config_name: Name of config file (e.g., 'benchmark_hyperparameters')
        experiment_key: Experiment identifier (e.g., 'resnet_cifar10')
        optimizer_name: Optimizer name (e.g., 'Adam')

    Returns:
        Dictionary with optimizer hyperparameters

    Example:
        >>> config = load_optimizer_config('benchmark_hyperparameters',
        ...                                'resnet_cifar10', 'Adam')
        >>> lr = config['lr']
        >>> optimizer = Adam(lr=lr, **config)
    """
    loader = get_config_loader()
    return loader.get_optimizer_config(config_name, experiment_key, optimizer_name)


def load_experiment_config(config_name: str, experiment_key: str) -> Dict[str, Any]:
    """
    Convenience function to load full experiment config.

    Args:
        config_name: Name of config file
        experiment_key: Experiment identifier

    Returns:
        Dictionary with all experiment settings
    """
    loader = get_config_loader()
    return loader.get_experiment_config(config_name, experiment_key)
