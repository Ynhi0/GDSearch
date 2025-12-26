import os
import json
from pathlib import Path
import pandas as pd
from scripts.generate_statistical_report import _load_final_metric


def test_load_final_metric_accepts_test_accuracy_alias(tmp_path, monkeypatch):
    # Create directories matching the expected pattern
    results_dir = tmp_path / "results"
    csv_dir = results_dir / "experiments" / "mnist" / "experiments" / "mnist"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Create a sample CSV with 'test_accuracy' column and seed in filename
    data = {
        'epoch': [1, 2],
        'test_accuracy': [0.85, 0.88]
    }
    df = pd.DataFrame(data)
    # Use a filename that matches the expected pattern in OPTIMIZER_PATTERNS
    fname = csv_dir / "MNIST_SimpleMLP_Adam_seed42.csv"
    df.to_csv(fname, index=False)

    # We need to ensure OPTIMIZER_PATTERNS contains the test key. Use 'Adam' which exists in default
    res = _load_final_metric(str(results_dir), 'Adam', 'test_acc')

    assert 42 in res
    assert abs(res[42] - 0.88) < 1e-6
