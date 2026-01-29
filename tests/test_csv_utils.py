import os
import pandas as pd

from src.utils.csv_utils import safe_read_csv


def test_safe_read_csv_empty(tmp_path, caplog):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    caplog.set_level("WARNING")
    res = safe_read_csv(str(empty))
    assert res is None
    assert any("is empty" in r.message for r in caplog.records)


def test_safe_read_csv_regular(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    res = safe_read_csv(str(p))
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (2, 2)


def test_cleanup_empty_csvs_moves_empty_and_unreadable_files(tmp_path):
    base = tmp_path
    good = base / "good.csv"
    empty = base / "bad_empty.csv"
    bad = base / "bad_parsable.csv"

    good.write_text('a,b\n1,2\n')
    empty.write_text('')
    # Create a file that will trigger pandas EmptyDataError when read with nrows=1
    # e.g., a file with only whitespace/newlines
    bad.write_text('\n\n')

    moved = cleanup_empty_csvs(str(base))
    # Expect two files moved (empty and bad)
    assert any('bad_empty.csv' in m for m in moved)
    assert any('bad_parsable.csv' in m for m in moved)
    # Good file remains
    assert good.exists()
    # Corrupt folder has the moved files
    corrupt = base / 'corrupt'
    assert (corrupt / 'bad_empty.csv').exists()
    assert (corrupt / 'bad_parsable.csv').exists()
