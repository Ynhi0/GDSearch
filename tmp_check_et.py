from src.core.experiment_tracker import ExperimentTracker
from pathlib import Path
import json
p = Path('tests/tmp_artifacts')
if p.exists():
    import shutil; shutil.rmtree(p)
p.mkdir(parents=True)
tr = ExperimentTracker(artifacts_dir=str(p), tracking_uri=None)
tr._write_resume_meta('run-xyz','checkpoints/a.pt')
print('wrote')
meta = tr._read_resume_meta()
print('meta:', meta)
# simulate active run and register checkpoint
class DummyRun:
    def __init__(self, rid):
        self.info = type('I',(object,),{'run_id':rid})
tr.current_run = DummyRun('run-abc')
tr.register_checkpoint('checkpoints/ckpt.pth')
print('after register:', json.load(open(p/'resume_meta.json')))
