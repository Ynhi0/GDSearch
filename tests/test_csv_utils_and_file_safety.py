import pandas as pd
import pytest
from pathlib import Path
import os
from src.utils.csv_utils import safe_read_csv, CSVReadError, cleanup_empty_csvs
from src.utils.file_safety import safe_to_csv, safe_write_text


def test_safe_read_csv_parser_error(tmp_path):
    p = tmp_path / "bad.csv"
    # Create an invalid CSV to trigger a parser error (unterminated quote)
    p.write_text('a,b\n"unterminated\n1,2')
    with pytest.raises(CSVReadError):
        safe_read_csv(p)


def test_safe_read_csv_empty(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert safe_read_csv(p) is None


def test_cleanup_empty_csvs_moves_empty(tmp_path):
    base = tmp_path
    empty = base / "empty.csv"
    empty.write_text("")
    moved = cleanup_empty_csvs(base)
    assert len(moved) == 1
    assert 'corrupt' in moved[0]


def test_safe_to_csv_propagates_io_error(monkeypatch, tmp_path):
    df = pd.DataFrame({'a': [1,2,3]})
    p = tmp_path / "path" / "out.csv"

    def raise_io(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, 'to_csv', raise_io)
    with pytest.raises(OSError):
        safe_to_csv(df, p)


def test_safe_write_text_propagates_io_error(monkeypatch, tmp_path):
    p = tmp_path / "out.txt"

    def raise_io(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, 'write_text', raise_io)
    with pytest.raises(OSError):
        safe_write_text("hello", p)
