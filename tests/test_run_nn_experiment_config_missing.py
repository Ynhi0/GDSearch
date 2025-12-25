import os
import pytest
import pandas as pd

from src.experiments import run_nn_experiment


def test_missing_config_raises_by_default(monkeypatch):
    # Simulate config missing
    monkeypatch.setattr(os.path, 'exists', lambda p: False)
    monkeypatch.delenv('GDSEARCH_ALLOW_DEFAULTS', raising=False)

    with pytest.raises(FileNotFoundError):
        run_nn_experiment.main()


def test_missing_config_allows_defaults_with_env(monkeypatch):
    # Simulate config missing and opt-in environment var
    monkeypatch.setattr(os.path, 'exists', lambda p: False)
    monkeypatch.setenv('GDSEARCH_ALLOW_DEFAULTS', '1')

    # Stub heavy operations to keep test fast and side-effect free
    monkeypatch.setattr(run_nn_experiment, 'train_and_evaluate', lambda cfg: pd.DataFrame([]))
    monkeypatch.setattr(run_nn_experiment, 'result_filename', lambda cfg: 'dummy.csv')
    monkeypatch.setattr(pd.DataFrame, 'to_csv', lambda self, path, index=True: None)
    monkeypatch.setattr(os, 'makedirs', lambda path, exist_ok=True: None)

    # Should not raise and should complete
    run_nn_experiment.main()
