import importlib
import builtins
import sys
import logging

import pytest


def test_experiment_tracker_import_failure_does_not_raise(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Simulate a runtime error while importing mlflow (e.g., MlflowException during import)
        if name == 'mlflow' or name.startswith('mlflow.'):
            raise RuntimeError("Simulated mlflow runtime import error")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    # Ensure module gets reloaded fresh
    if 'src.core.experiment_tracker' in sys.modules:
        del sys.modules['src.core.experiment_tracker']

    # Import should not raise despite mlflow import failure
    mod = importlib.import_module('src.core.experiment_tracker')
    assert getattr(mod, 'HAS_MLFLOW', False) is False
    assert any('mlflow import failed' in rec.message for rec in caplog.records)

    # restore import behavior implicitly by fixture teardown
