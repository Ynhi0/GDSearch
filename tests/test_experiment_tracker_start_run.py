from types import SimpleNamespace
import src.core.experiment_tracker as et


def test_start_run_handles_missing_info(monkeypatch):
    # Create a fake mlflow with set_experiment/start_run
    fake_mlflow = SimpleNamespace()

    def fake_set_experiment(name):
        return None

    def fake_start_run(run_name=None, nested=False):
        # Return object without 'info' attribute to simulate edge case
        return SimpleNamespace()

    fake_mlflow.set_experiment = fake_set_experiment
    fake_mlflow.start_run = fake_start_run

    monkeypatch.setattr(et, 'mlflow', fake_mlflow)
    monkeypatch.setattr(et, 'HAS_MLFLOW', True)

    tracker = et.ExperimentTracker()
    # tracker should enable mlflow integration
    assert tracker.enabled

    run_id = tracker.start_run(run_name="test")
    # Should not raise and should return None when info/run_id missing
    assert run_id is None
