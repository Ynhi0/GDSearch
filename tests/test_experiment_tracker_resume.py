import json
import os
import types
import tempfile
import pytest
from src.core.experiment_tracker import ExperimentTracker


def test_write_and_read_resume_meta(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    et = ExperimentTracker(artifacts_dir=str(art), tracking_uri=None)
    # Exercise private API (small unit test)
    et._write_resume_meta("run-123", "checkpoints/ckpt.pth")
    meta = et._read_resume_meta()
    assert meta["run_id"] == "run-123"
    assert meta["checkpoint"].endswith("ckpt.pth")


def test_start_run_resumes_when_meta_present(monkeypatch, tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    et = ExperimentTracker(artifacts_dir=str(art), tracking_uri=None)
    et._write_resume_meta("run-123", "checkpoints/ckpt.pth")

    called = {}

    class DummyRun:
        def __init__(self, run_id):
            self.info = types.SimpleNamespace(run_id=run_id)

    def fake_start_run(**kw):
        # emulate mlflow.start_run(run_id=...)
        run_id = kw.get('run_id') or kw.get('run_name') or 'new'
        called['run_id'] = run_id
        return DummyRun(run_id)

    monkeypatch.setattr('src.core.experiment_tracker.mlflow.start_run', fake_start_run)
    # Should attach to persisted run when resume=True
    rid = et.start_run(resume=True)
    assert called['run_id'] == 'run-123'
    assert rid == 'run-123'


def test_register_checkpoint_updates_meta(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    et = ExperimentTracker(artifacts_dir=str(art), tracking_uri=None)
    # Simulate an active run by monkeypatching current_run.info
    class DummyRun:
        def __init__(self, run_id):
            self.info = types.SimpleNamespace(run_id=run_id)

    et.current_run = DummyRun('run-abc')
    et.register_checkpoint('checkpoints/ckpt.pth')
    meta = json.load(open(os.path.join(str(art), "resume_meta.json")))
    assert meta['run_id'] == 'run-abc'
    assert 'ckpt.pth' in meta['checkpoint']


def test_start_run_persists_meta(monkeypatch, tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    et = ExperimentTracker(artifacts_dir=str(art), tracking_uri=None)

    class DummyRun:
        def __init__(self, run_id):
            self.info = types.SimpleNamespace(run_id=run_id)

    called = {}
    def fake_start_run(**kw):
        called['kw'] = kw
        return DummyRun(kw.get('run_id') or kw.get('run_name') or 'new')

    monkeypatch.setattr('src.core.experiment_tracker.mlflow.start_run', fake_start_run)
    rid = et.start_run(run_name='myrun', resume=False)
    # resume_meta should have been written with run_id
    meta = json.load(open(os.path.join(str(art), "resume_meta.json")))
    assert meta['run_id'] == rid
    assert meta['checkpoint'] is None