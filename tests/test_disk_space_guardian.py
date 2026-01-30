import os
import time
from pathlib import Path
from src.core.training_enhancements import DiskSpaceGuardian


def touch_with_mtime(path: Path, mtime: float):
    path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_cleanup_deletes_oldest(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # Create 6 fake checkpoints with increasing mtime
    now = time.time()
    files = []
    for i in range(6):
        p = ckpt_dir / f"ckpt_{i}.pt"
        touch_with_mtime(p, now + i)
        files.append(p)

    guardian = DiskSpaceGuardian(ckpt_dir, max_checkpoints=3)
    guardian._cleanup_old_checkpoints()

    remaining = sorted(ckpt_dir.glob("*.pt"))
    # Should keep the 3 newest files: ckpt_3, ckpt_4, ckpt_5
    remaining_names = sorted(p.name for p in remaining)
    assert len(remaining_names) == 3
    assert remaining_names == ["ckpt_3.pt", "ckpt_4.pt", "ckpt_5.pt"]


def test_cleanup_no_delete_when_under_limit(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    p = ckpt_dir / "only.pt"
    touch_with_mtime(p, time.time())

    guardian = DiskSpaceGuardian(ckpt_dir, max_checkpoints=5)
    guardian._cleanup_old_checkpoints()

    remaining = list(ckpt_dir.glob("*.pt"))
    assert len(remaining) == 1
    assert remaining[0].name == "only.pt"
