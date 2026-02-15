import logging, time, os, sys
from pathlib import Path
# Ensure project src is on path for ad-hoc debug runs
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from src.core.checkpoint_manager import RobustCheckpointManager

logging.basicConfig(level=logging.DEBUG)
base = Path(r'C:/Users/MPhuc/AppData/Local/Temp/test_stale_debug')
if base.exists():
    import shutil
    shutil.rmtree(base)
base.mkdir(parents=True)
ckpt = base / 'model.pt'
ckpt.write_text('x')
lock = base / 'model.pt.backup.lock'
lock.write_text('stale')
old = time.time() - 7200
os.utime(lock, (old, old))
print('Before: lock exists', lock.exists())
mgr = RobustCheckpointManager(str(base), max_backups=2)
try:
    mgr._create_backup(ckpt, 'exp')
except Exception as e:
    print('Exception raised:', e)
print('After: backup exists', (base / 'model.pt.backup_0').exists())
print('Lock exists:', lock.exists())
# Try to unlink lock manually to check if unlinking is possible
try:
    lock.unlink()
    print('Manual unlink succeeded; lock removed')
except Exception as e:
    print('Manual unlink failed:', type(e), e)
print('Lock still exists?:', lock.exists())
