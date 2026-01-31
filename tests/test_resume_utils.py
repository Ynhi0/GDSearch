import tempfile
from pathlib import Path
import pandas as pd
from src.core.resume_utils import results_exist, compute_run_signature


def test_results_exist_with_corrupt_csv(tmp_path, caplog):
    # Create a corrupt CSV (binary content) and ensure results_exist returns False
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    summary = results_dir / "summary_quantitative.csv"
    summary.write_bytes(b"\x00\x01\x02not a csv")

    assert results_exist(results_dir, "doesnotmatter") is False


def test_compute_run_signature_handles_unserializable_values():
    config = {"a": 1, "b": {"x": 2}, "c": {1, 2, 3}}  # set is not JSON serializable
    sig = compute_run_signature(config)
    assert isinstance(sig, str)
    assert len(sig) == 64  # sha256 hex length
