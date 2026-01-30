import threading
import time
from pathlib import Path
from src.core.checkpoint_manager import RobustCheckpointManager


def worker(mgr, ckpt, errs):
    try:
        mgr._create_backup(ckpt, 'exp')
    except Exception as e:
        errs.append(e)


def test_concurrent_backup_lock_only_one_creator(tmp_path):
    base = tmp_path
    mgr = RobustCheckpointManager(str(base), max_backups=1, backup_lock_timeout=3, stale_lock_seconds=2)

    ckpt = base / 'model.pt'
    ckpt.write_text('original-data')

    threads = []
    errors = []

    for _ in range(5):
        t = threading.Thread(target=worker, args=(mgr, ckpt, errors))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Ensure no thread raised an unexpected exception
    assert not errors, f"Unexpected errors in worker threads: {errors}"

    # Backup must exist and match the original
    backup = base / 'model.pt.backup_0'
    assert backup.exists(), "Backup was not created by any thread"
    assert backup.read_text() == 'original-data'

    # No leftover lock should remain
    assert not (base / 'model.pt.backup.lock').exists(), "Lock file left behind"
