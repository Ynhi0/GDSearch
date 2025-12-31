"""
Simple checkpoint roundtrip test used by CI. Creates a RobustCheckpointManager,
saves a small checkpoint, loads it back, and asserts consistency.
"""
import sys
from pathlib import Path
import tempfile

from src.core.checkpoint_manager import RobustCheckpointManager

with tempfile.TemporaryDirectory() as td:
    mgr = RobustCheckpointManager(td, strict=True)
    data = {'model_state_dict': {'dummy': 1}, 'meta': {'test': True}}
    filename = 'ci_test.ckpt'
    try:
        ok = mgr.save_checkpoint(data, filename=filename, experiment_name='ci_test')
    except Exception as e:
        print("Checkpoint save failed:", e)
        sys.exit(2)

    if not ok:
        print("Checkpoint manager returned False during save (strict mode expected to raise on error)")
        sys.exit(2)

    ckpt = mgr.load_checkpoint(filename)
    if ckpt is None:
        print("Failed to load checkpoint back")
        sys.exit(2)

    # Basic validation
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        print("Loaded checkpoint missing expected keys")
        sys.exit(2)

print("Checkpoint roundtrip OK")
sys.exit(0)
