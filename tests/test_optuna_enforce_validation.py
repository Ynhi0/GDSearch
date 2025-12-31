import pytest

try:
    from src.core.optuna_tuner import OptunaHyperparameterTuner
except Exception:
    OptunaHyperparameterTuner = None


def test_optuna_requires_validation_loader():
    if OptunaHyperparameterTuner is None:
        pytest.skip("Optuna not available in this environment")

    def objective(trial):
        return 0.0

    tuner = OptunaHyperparameterTuner(objective, seed=42)
    with pytest.raises(ValueError):
        tuner.optimize(n_trials=1, val_loader=None, enforce_validation=True)
