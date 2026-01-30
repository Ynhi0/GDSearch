import os
import time
from pathlib import Path
from src.core.checkpoint_manager import RobustCheckpointManager


def test_token_prevents_unintended_unlock(tmp_path):
    base = tmp_path
    mgr = RobustCheckpointManager(str(base), max_backups=2, backup_lock_timeout=1, stale_lock_seconds=3600)

    ckpt = base / 'model.pt'
    ckpt.write_text('original')

    # Create a lock file with a different token and recent mtime (not stale)
    lock = base / 'model.pt.backup.lock'
    lock.write_text('99999:deadbeef')
    # Ensure mtime is now (recent) so it is NOT considered stale
    now = time.time()
    os.utime(lock, (now, now))

    # Attempt to create a backup - it should timeout and NOT remove the lock
    mgr._create_backup(ckpt, 'exp')

    # Lock should still exist and backup should not be created
    assert lock.exists(), "Lock file was removed by someone without matching token"
    assert not (base / 'model.pt.backup_0').exists(), "Backup unexpectedly created while lock held by different token"
