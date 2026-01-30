import pytest
import logging

import run_all_kaggle as runner


def test_create_experiment_tracker_handles_exceptions(monkeypatch):
    class BadTrainer:
        def __init__(self):
            raise RuntimeError("boom: schema is out-of-date")

    monkeypatch.setattr(runner, 'ExperimentTracker', BadTrainer)
    tracker = runner._create_experiment_tracker(no_mlflow=False)
    assert tracker is None


def test_create_experiment_tracker_disabled_when_no_mlflow(monkeypatch):
    # Simulate HAS_MLFLOW False
    monkeypatch.setattr(runner, 'HAS_MLFLOW', False)
    tracker = runner._create_experiment_tracker(no_mlflow=False)
    assert tracker is None

    tracker = runner._create_experiment_tracker(no_mlflow=True)
    assert tracker is None
