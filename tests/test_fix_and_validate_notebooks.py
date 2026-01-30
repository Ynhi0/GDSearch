import json
from pathlib import Path
import pytest
nbformat = pytest.importorskip('nbformat')

from scripts.validate_notebooks import check_notebook
from scripts.fix_and_validate_notebooks import validate_notebook_structure


def make_notebook_with_literal_newline(tmp_path):
    nb = nbformat.v4.new_notebook()
    # Create a code cell that incorrectly contains a literal '\n' sequence
    src = "from src.utils.csv_utils import safe_read_csv\\nmnist_csvs = list((RESULTS_DIR / 'experiments' / 'mnist').glob('*.csv'))"
    cell = nbformat.v4.new_code_cell(src)
    nb.cells.append(cell)
    nb_path = tmp_path / 'bad.ipynb'
    with nb_path.open('w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    return nb_path


def test_fix_and_validate(tmp_path):
    nb_path = make_notebook_with_literal_newline(tmp_path)
    # Run check_notebook with fix enabled
    changed = check_notebook(nb_path, fix=True)
    assert changed
    # Validate structure
    assert validate_notebook_structure(nb_path)
    # Read notebook and ensure the code cell now contains a true newline
    nb = nbformat.read(str(nb_path), as_version=4)
    src = nb.cells[0].source
    assert '\\n' not in src
    assert 'safe_read_csv' in src and 'mnist_csvs' in src
