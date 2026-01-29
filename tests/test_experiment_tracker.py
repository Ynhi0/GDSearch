import pytest
from unittest import mock

from src.core import experiment_tracker


def test_init_with_mlflow_failure(monkeypatch):
    # Simulate mlflow is installed but initialization fails (e.g. DB schema error)
    class DummyMlflow:
        def set_tracking_uri(self, uri):
            pass

        def set_experiment(self, name):
            raise RuntimeError("simulated mlflow schema error")

    monkeypatch.setattr(experiment_tracker, "HAS_MLFLOW", True)
    monkeypatch.setattr(experiment_tracker, "mlflow", DummyMlflow())

    # Should not raise and should set enabled = False
    tracker = experiment_tracker.ExperimentTracker()
    assert getattr(tracker, "enabled", False) is False


def test_methods_noop_when_disabled():
    tracker = experiment_tracker.ExperimentTracker()
    tracker.enabled = False

    # Methods should act as no-ops and not raise
    assert tracker.start_run() is None
    tracker.log_params({"a": 1})
    tracker.log_metrics({"m": 0.1})
    tracker.end_run()
    tracker.log_artifact("/tmp/nonexistent")
    tracker.log_model(mock.Mock())
