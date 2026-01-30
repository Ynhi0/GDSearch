import os
import time
from pathlib import Path
from src.core.checkpoint_manager import RobustCheckpointManager


def test_stale_lock_unlink_permission_error(tmp_path, monkeypatch):
    base = tmp_path
    mgr = RobustCheckpointManager(str(base), max_backups=1)

    ckpt = base / 'model.pt'
    ckpt.write_text('x')

    # Create a stale lock file older than 2 hours
    lock = base / 'model.pt.backup.lock'
    lock.write_text('stale')
    old = time.time() - 7200
    os.utime(lock, (old, old))

    # Monkeypatch unlink to raise PermissionError for the stale lock
    orig_unlink = Path.unlink

    def fake_unlink(self, *args, **kwargs):
        if self == lock:
            raise PermissionError("permission denied for unlink")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fake_unlink)

    # Should not raise despite unlink failure; backup should still be attempted
    mgr._create_backup(ckpt, 'exp')

    # Backup 0 should exist (copy may succeed even if stale lock couldn't be removed)
    assert (base / 'model.pt.backup_0').exists()

    # The stale lock file may still be present due to unlink failure
    assert lock.exists(), "Expected stale lock to remain when unlink fails"


def test_release_lock_unlink_permission_error(tmp_path, monkeypatch):
    base = tmp_path
    mgr = RobustCheckpointManager(str(base), max_backups=1)

    ckpt = base / 'model.pt'
    ckpt.write_text('original-data')

    lock = base / 'model.pt.backup.lock'

    # Monkeypatch unlink to raise PermissionError only for the lock file
    orig_unlink = Path.unlink

    def fake_unlink(self, *args, **kwargs):
        if self == lock:
            raise PermissionError("cannot unlink lock on release")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fake_unlink)

    # Should not raise despite unlink failure at release
    mgr._create_backup(ckpt, 'exp')

    # Backup must exist
    backup = base / 'model.pt.backup_0'
    assert backup.exists(), "Backup was not created"

    # Because unlink failed at release, lock file may remain
    assert lock.exists(), "Expected lock file to remain when unlink on release fails"