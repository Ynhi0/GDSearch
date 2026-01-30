import inspect
from pathlib import Path


def test_run_all_uses_safe_read_csv():
    p = Path('run_all_kaggle.py')
    text = p.read_text()
    assert 'safe_read_csv(' in text, "run_all_kaggle.py should use safe_read_csv for external CSV reads"
