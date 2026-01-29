import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.visualization.plotting_utils import filter_time_series_files, safe_add_text, normalize_final_results


def test_filter_time_series_files(tmp_path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text('epoch,train_loss\n1,0.5\n')
    f2.write_text('final_test_acc\n0.9\n')

    res = filter_time_series_files([f1, f2])
    assert f1 in res
    assert f2 not in res


def test_safe_add_text():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # finite coords -> should not raise
    safe_add_text(ax, 1.0, 2.0, 'ok')
    # non-finite -> does nothing
    safe_add_text(ax, float('nan'), 2.0, 'nope')
    safe_add_text(ax, 1.0, float('inf'), 'nope')
    plt.close(fig)


def test_normalize_final_results():
    s = pd.Series({'Adam': 0.92, 'SGD': 92.0})
    df = normalize_final_results(s)
    assert 'Adam' in df.index
    assert df['mean'].max() <= 100.0
    assert df['mean'].min() >= 0.0
