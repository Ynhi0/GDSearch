from pathlib import Path
from run_all_kaggle import save_run_artifacts
import json

p = Path('tests/tmp_save_artifacts')
import shutil
shutil.rmtree(p, ignore_errors=True)
p.mkdir(parents=True)

csv, meta = save_run_artifacts(str(p), 'MNIST', 'SimpleMLP', 'SGD', 1001, [], {'lr': 0.01})
print('csv:', csv)
print('meta:', meta)
print('csv exists:', (Path(csv).exists() if csv else None))
print('csv size:', (Path(csv).stat().st_size if csv else None))
if meta:
    with open(meta, 'r', encoding='utf-8') as f:
        print('meta content:', json.load(f))
