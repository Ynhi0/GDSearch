import importlib
import logging
import pandas as pd
import types

import pytest

from src.core import experiment_tracker
import run_all_kaggle


def _raise_schema_error(*args, **kwargs):
    raise Exception("Simulated mlflow DB schema error")


def test_experiment_tracker_handles_set_experiment_error(monkeypatch, caplog):
    """If mlflow.set_experiment() raises (DB schema error), ExperimentTracker should disable itself and not raise."""
    caplog.set_level(logging.WARNING)

    # Ensure mlflow module has set_experiment we can patch; if mlflow missing, provide a dummy module
    if getattr(experiment_tracker, "mlflow", None) is None:
        dummy = types.SimpleNamespace()
        dummy.set_experiment = _raise_schema_error
        monkeypatch.setattr(experiment_tracker, "mlflow", dummy, raising=False)
    else:
        monkeypatch.setattr(experiment_tracker.mlflow, "set_experiment", _raise_schema_error)

    # Create tracker - should not raise, and should be disabled
    tracker = experiment_tracker.ExperimentTracker(experiment_name="test_fail_init")
    assert tracker.enabled is False
    # Confirm warning was logged
    assert any("MLflow initialization failed" in rec.message or "ExperimentTracker created but not enabled" in rec.message for rec in caplog.records)


def test_run_all_kaggle_main_continues_when_mlflow_init_fails(monkeypatch, caplog, capsys):
    """Simulate mlflow DB schema error and verify run_all_kaggle.main() completes without raising and logs an informative warning."""
    caplog.set_level(logging.WARNING)

    # Patch experiment_tracker.mlflow.set_experiment to raise
    if getattr(experiment_tracker, "mlflow", None) is None:
        dummy = types.SimpleNamespace()
        dummy.set_experiment = _raise_schema_error
        monkeypatch.setattr(experiment_tracker, "mlflow", dummy, raising=False)
    else:
        monkeypatch.setattr(experiment_tracker.mlflow, "set_experiment", _raise_schema_error)

    # Ensure run_all_kaggle thinks mlflow package is available so it will attempt to create ExperimentTracker
    monkeypatch.setattr(run_all_kaggle, "HAS_MLFLOW", True)

    # Monkeypatch heavy experiment functions to be fast no-ops to allow main() to complete quickly
    def fake_mnist(*args, **kwargs):
        # Return an empty DataFrame as a fast stand-in
        return pd.DataFrame([])

    monkeypatch.setattr(run_all_kaggle, "run_mnist_experiment", fake_mnist)

    # Run main with minimal, quick arguments. Simulate CLI args to do only the MNIST experiment in ultra-quick mode
    monkeypatch.setattr("sys.argv", ["run_all_kaggle.py", "--ultra-quick", "--experiments", "mnist"])

    # Run main and assert it does not raise
    run_all_kaggle.main()

    # Check logs for our decisive warning about MLflow failures
    logged = "\n".join(rec.message for rec in caplog.records)
    assert ("MLflow initialization failed" in logged) or ("ExperimentTracker created but not enabled" in logged) or ("Disabling tracker for this run" in logged)

    # Also assert that the printed MLflow status reflects the failure (either 'failed' or 'FAILED')
    out = capsys.readouterr().out
    assert ("MLflow: failed" in out.lower()) or ("mlflow tracking: failed" in out.lower()) or ("mlflow tracking: disabled" in out.lower())
