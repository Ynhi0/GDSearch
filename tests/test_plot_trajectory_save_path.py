import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from src.visualization.plot_results import plot_trajectory_and_step_size
from src.core.test_functions import Rosenbrock


def test_plot_trajectory_and_step_size_save_path(tmp_path):
    # Create a small synthetic trajectory dataframe with required columns
    df = pd.DataFrame({'x': np.linspace(-1.5, -1.0, 10), 'y': np.linspace(2.0, 1.0, 10)})
    df['iteration'] = np.arange(len(df))
    df['step_size'] = np.linspace(0.1, 0.01, len(df))

    out_file = tmp_path / 'traj_step.png'

    # Should not raise and should create the file
    plot_trajectory_and_step_size(df, Rosenbrock(), title='test', save_path=str(out_file))
    assert out_file.exists(), f"Expected saved plot at {out_file}"
