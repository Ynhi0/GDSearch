import os
import time
from pathlib import Path
from run_all_kaggle import RobustCheckpointManager


def test_stale_lock_is_removed_and_backup_created(tmp_path):
    base = tmp_path
    mgr = RobustCheckpointManager(str(base), max_backups=2)

    ckpt = base / 'model.pt'
    ckpt.write_text('x')

    # Create a stale lock file older than 2 hours
    lock = base / 'model.pt.backup.lock'
    lock.write_text('stale')
    old = time.time() - 7200
    os.utime(lock, (old, old))

    mgr._create_backup(ckpt, 'exp')

    # Backup 0 should exist now
    assert (base / 'model.pt.backup_0').exists()
