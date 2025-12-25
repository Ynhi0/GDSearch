import os
import pytest
from src.core.optuna_tuner import create_tuner, RandomTuner, OptunaHyperparameterTuner


def dummy_objective(trial):
    # Objective expects trial.suggest_float('x', ...) and returns x**2
    x = trial.suggest_float('x', -1.0, 1.0)
    return x * x


def test_default_returns_random_tuner(monkeypatch):
    monkeypatch.delenv('GDSEARCH_ENABLE_OPTUNA', raising=False)
    tuner = create_tuner(dummy_objective)
    assert isinstance(tuner, RandomTuner)


def test_env_enables_optuna_if_available(monkeypatch):
    monkeypatch.setenv('GDSEARCH_ENABLE_OPTUNA', '1')

    try:
        # If optuna is installed and working, create_tuner should return OptunaHyperparameterTuner
        tuner = create_tuner(dummy_objective)
        assert isinstance(tuner, OptunaHyperparameterTuner)
    except RuntimeError:
        # If optuna not installed, we expect a RuntimeError telling how to install
        with pytest.raises(RuntimeError):
            create_tuner(dummy_objective, use_optuna=True)


def test_force_disable_with_param():
    tuner = create_tuner(dummy_objective, use_optuna=False)
    assert isinstance(tuner, RandomTuner)


def test_random_tuner_runs_trials():
    tuner = create_tuner(dummy_objective, use_optuna=False)
    results = tuner.optimize(n_trials=5)
    assert 'best_params' in results
    assert results['n_trials'] > 0
