"""
Default hyperparameters for optimizers across different experiment types.

This module provides a centralized source of truth for hyperparameter defaults
and configuration loading.
"""
# broad catch intentional - this module may catch broadly when reading optional
# config artifacts at runtime; failures are logged and sensible defaults are used
import json
import logging
from pathlib import Path
from typing import Dict


def get_default_hyperparameters(
    optimizer_name: str,
    experiment_type: str = "2d_optimization"
) -> Dict:
    """
    Get default hyperparameters from tuned config file.

    Args:
        optimizer_name: Name of optimizer (e.g., 'Adam', 'SGD', 'AdamW')
        experiment_type: Type of experiment (e.g., '2d_optimization', 'mnist', 'cifar10')

    Returns:
        Dictionary of hyperparameters
    """
    try:
        # Try to load from config file
        config_path = Path(__file__).parent.parent.parent / "configs" / "benchmark_hyperparameters.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get hyperparameters for the specific experiment type
            exp_config = config.get("experiment_configs", {}).get(experiment_type, {})
            optimizers_cfg = exp_config.get("optimizers", {})

            # Normalize optimizer names from config to canonical keys
            from src.core.optimizer_registry import normalize_optimizer_name

            normalized_cfg = {}
            for k, v in optimizers_cfg.items():
                try:
                    canon = normalize_optimizer_name(k)
                    normalized_cfg[canon] = v
                except ValueError as e:
                    # Fail-fast: unknown optimizer specified in config
                    raise RuntimeError(f"Invalid optimizer name in hyperparameter config: '{k}': {e}") from e

            # Normalize requested optimizer name
            try:
                canonical_name = normalize_optimizer_name(optimizer_name)
            except ValueError:
                canonical_name = optimizer_name  # use as-is and allow fallback

            opt_config = normalized_cfg.get(canonical_name, None)

            if opt_config is not None:
                return opt_config
            else:
                logging.warning(
                    "No hyperparameters found in config for optimizer '%s' (normalized: '%s'). Using fallback defaults.",
                    optimizer_name, canonical_name
                )
    except Exception as e:
        logging.warning(
            "Could not load hyperparameters from config: %s, using fallback defaults",
            e
        )

    # Fallback defaults if config loading fails
    defaults = {
        'SGD': {'lr': 0.01},
        # Align default with README and benchmark config: SGD momentum lr=0.01
        'SGD_Momentum': {'lr': 0.01, 'momentum': 0.9},
        'Adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
        'AdamW': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 1e-4},
        'AMSGrad': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
        'SAM_SGD': {'lr': 0.01, 'rho': 0.05},
        'SAM_Adam': {'lr': 0.001, 'rho': 0.05},
        'Lookahead_SGD': {'lr': 0.01, 'k': 5, 'alpha': 0.5},
        'Lookahead_Adam': {'lr': 0.001, 'k': 5, 'alpha': 0.5},
        'AdaBound': {
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'final_lr': 0.1,
            'gamma': 1e-3
        },
        'RAdam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
        'LAMB': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0.01}
    }
    return defaults.get(optimizer_name, {'lr': 0.001})
