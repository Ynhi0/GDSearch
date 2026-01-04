import os
import pandas as pd
import numpy as np

from src.core.test_functions import Rosenbrock
from src.experiments.run_optimizer_ablation import run_optimizer_ablation


def test_run_optimizer_ablation_outputs(tmp_path):
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    rosen = Rosenbrock(a=1, b=100)

    df = run_optimizer_ablation(
        test_function=rosen,
        initial_point=(0.0, 0.0),
        max_iterations=10,
        results_dir=str(results_dir),
        plots_dir=str(plots_dir)
    )

    # Basic structure
    assert isinstance(df, pd.DataFrame)
    assert 'Optimizer' in df.columns
    assert 'LR' in df.columns
    assert len(df) == 6  # 6 optimizers expected

    expected_lr = {
        'SGD': 0.1,
        'SGD+Momentum': 0.1,
        'RMSProp': 0.001,
        'Adam': 0.001,
        'AdamW': 0.001,
        'AMSGrad': 0.001,
    }

    # Verify LR mapping approximately matches expected defaults
    for _, row in df.iterrows():
        opt = str(row['Optimizer'])
        lr = float(row['LR'])
        assert np.isclose(lr, expected_lr[opt], rtol=1e-2) or np.isclose(lr, 0.01, rtol=1e-2)  # allow legacy fallback

    # Check summary CSV exists
    summary_csv = results_dir / 'optimizer_ablation_summary.csv'
    assert summary_csv.exists()
