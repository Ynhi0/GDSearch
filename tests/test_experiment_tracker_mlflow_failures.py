import importlib
import sys
import types
import pytest


def _make_fake_mlflow(raise_on_set=False, raise_on_start=False):
    # Create fake mlflow module with optional exceptions
    mlflow = types.SimpleNamespace()

    class MlflowException(Exception):
        pass

    mlflow.exceptions = types.SimpleNamespace(MlflowException=MlflowException)

    def set_experiment(name):
        if raise_on_set:
            raise MlflowException("db init failed")
        return None

    def set_tracking_uri(uri):
        if uri == "bad://":
            raise RuntimeError("bad uri")

    def start_run(run_name=None, nested=False):
        if raise_on_start:
            raise MlflowException("start failed")
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="fake"))

    def end_run():
        return None

    mlflow.set_experiment = set_experiment
    mlflow.set_tracking_uri = set_tracking_uri
    mlflow.start_run = start_run
    mlflow.end_run = end_run

    # provide a simple mlflow.pytorch
    mlflow.pytorch = types.SimpleNamespace(log_model=lambda m, n: None)

    return mlflow


def test_init_handles_mlflow_initialization_failure(monkeypatch):
    fake_mlflow = _make_fake_mlflow(raise_on_set=True)
    monkeypatch.setitem(sys.modules, 'mlflow', fake_mlflow)

    # reload the module to pick up our fake mlflow
    import src.core.experiment_tracker as et
    importlib.reload(et)

    tr = et.ExperimentTracker(experiment_name="foo")
    assert tr.enabled is False


def test_start_run_propagates_mlflow_exception_and_restores_stack(monkeypatch):
    fake_mlflow = _make_fake_mlflow(raise_on_start=True)
    monkeypatch.setitem(sys.modules, 'mlflow', fake_mlflow)

    import src.core.experiment_tracker as et
    importlib.reload(et)

    tr = et.ExperimentTracker()
    # Force enabled by bypassing init (mlflow fake set_experiment doesn't raise now)
    tr.enabled = True
    tr.current_run = "parent"
    tr.run_stack = []

    with pytest.raises(Exception):
        tr.start_run(run_name="child")

    # Ensure the stack was restored
    assert tr.run_stack == []


def test_log_param_handles_mlflow_domain_error(monkeypatch, caplog):
    fake_mlflow = _make_fake_mlflow()

    def bad_log_param(k, v):
        raise fake_mlflow.exceptions.MlflowException("log failed")

    fake_mlflow.log_param = bad_log_param
    monkeypatch.setitem(sys.modules, 'mlflow', fake_mlflow)

    import src.core.experiment_tracker as et
    importlib.reload(et)

    tr = et.ExperimentTracker()
    tr.enabled = True
    tr.current_run = True

    tr.log_params({'a': 1})
    assert any("Failed to log param" in r.message for r in caplog.records) or any("log failed" in r.message for r in caplog.records)
