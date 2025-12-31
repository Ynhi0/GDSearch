import os
from src.core.checkpoint_manager import RobustCheckpointManager


def test_save_and_load_checkpoint(tmp_path):
    base_dir = tmp_path
    manager = RobustCheckpointManager(str(base_dir))

    data = {'a': 1, 'b': 'test'}
    filename = 'test.ckpt'

    # Save should succeed
    ok = manager.save_checkpoint(data, filename, experiment_name='unit_test')
    assert ok, "save_checkpoint should return True"

    # The file should exist
    ckpt_path = base_dir / filename
    assert ckpt_path.exists(), "checkpoint file should exist after save"

    # Load should return a dict with the same keys
    loaded = manager.load_checkpoint(filename)
    assert isinstance(loaded, dict), "loaded checkpoint should be a dict"
    assert 'a' in loaded and loaded['a'] == 1
    assert 'b' in loaded and loaded['b'] == 'test'