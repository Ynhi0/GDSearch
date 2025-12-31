import pytest
from pathlib import Path
import tempfile

from src.core.checkpoint_manager import RobustCheckpointManager


def test_save_checkpoint_raises_on_atomic_failure(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        mgr = RobustCheckpointManager(td, strict=True)

        # Simulate atomic save failure by making torch_save_safe raise
        def fake_save(obj, path_or_file, use_new_zipfile_serialization=True):
            raise OSError("simulated disk full")

        monkeypatch.setattr('src.core.checkpoint_manager.torch_save_safe', fake_save)

        with pytest.raises(RuntimeError):
            mgr.save_checkpoint({'model': {}}, filename='tmp.ckpt', experiment_name='test')


def test_save_checkpoint_returns_false_in_non_strict(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        mgr = RobustCheckpointManager(td, strict=False)

        def fake_save(obj, path_or_file, use_new_zipfile_serialization=True):
            raise OSError("simulated disk full")

        monkeypatch.setattr('src.core.checkpoint_manager.torch_save_safe', fake_save)

        ok = mgr.save_checkpoint({'model': {}}, filename='tmp.ckpt', experiment_name='test')
        assert ok is False
