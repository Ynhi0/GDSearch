import pytest

from src.core.optimizer_registry import normalize_optimizer_name
from src.core.hyperparameters import get_default_hyperparameters
from typing import Any, cast


def test_normalize_aliases():
    assert normalize_optimizer_name('SGDMomentum') == 'SGD_Momentum'
    assert normalize_optimizer_name('sgdmomentum') == 'SGD_Momentum'
    assert normalize_optimizer_name('SGD_Momentum') == 'SGD_Momentum'
    assert normalize_optimizer_name('adamw') == 'AdamW'


def test_get_default_hyperparameters_normalized_lookup():
    # This relies on configs/benchmark_hyperparameters.json containing '2d_optimization' -> 'SGDMomentum'
    params = get_default_hyperparameters('SGDMomentum', experiment_type='2d_optimization')
    # Expect a dict with 'lr' key
    assert isinstance(params, dict)
    assert 'lr' in params


def test_unknown_optimizer_in_config_raises():
    # Ensure unknown optimizer name in normalization raises
    with pytest.raises(ValueError):
        normalize_optimizer_name(cast(Any, 123))  # non-string (runtime error expected)

    with pytest.raises(ValueError):
        normalize_optimizer_name('NotAnOptimizer')


def test_fallback_default_matches_readme():
    # Use a non-existent experiment type to trigger fallback defaults
    params = get_default_hyperparameters('SGD_Momentum', experiment_type='this_experiment_does_not_exist')
    assert isinstance(params, dict)
    assert params.get('lr') == 0.01
