from pathlib import Path
from src.utils.csv_utils import cleanup_empty_csvs
import shutil

p = Path('tests/tmp_csv_test')
shutil.rmtree(p, ignore_errors=True)
p.mkdir()
(p/'good.csv').write_text('a,b\n1,2\n')
(p/'bad.csv').write_text('')

moved = cleanup_empty_csvs(str(p))
print('moved:', moved)
print('good exists:', (p/'good.csv').exists())
print('corrupt exists:', (p/'corrupt'/'bad.csv').exists())
