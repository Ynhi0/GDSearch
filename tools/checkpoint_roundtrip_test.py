"""
Simple checkpoint roundtrip test used by CI. Converts to a pytest test that
creates a RobustCheckpointManager, saves a small checkpoint, loads it back,
and asserts consistency.
"""
from pathlib import Path

import pytest

from src.core.checkpoint_manager import RobustCheckpointManager


def test_checkpoint_roundtrip(tmp_path: Path):
    """Save and load a small checkpoint using RobustCheckpointManager."""
    mgr = RobustCheckpointManager(tmp_path, strict=True)
    data = {"model_state_dict": {"dummy": 1}, "meta": {"test": True}}
    filename = "ci_test.ckpt"

    # Save should succeed and return truthy
    ok = mgr.save_checkpoint(data, filename=filename, experiment_name="ci_test")
    assert ok, "Checkpoint manager failed to save checkpoint in strict mode"

    # Load back
    ckpt = mgr.load_checkpoint(filename)
    assert ckpt is not None, "Failed to load checkpoint back"

    # Basic validation
    assert isinstance(ckpt, dict), "Loaded checkpoint must be a dict"
    assert "model_state_dict" in ckpt, "Loaded checkpoint missing expected key 'model_state_dict'"
