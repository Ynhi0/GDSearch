import os
import pandas as pd
from src.experiments.run_optimizer_ablation import run_optimizer_ablation
from src.core.test_functions import Rosenbrock


def test_run_optimizer_ablation_uses_fair_lrs(tmp_path):
    results_dir = str(tmp_path / "results")
    plots_dir = str(tmp_path / "plots")

    rosen = Rosenbrock(a=1, b=100)
    df = run_optimizer_ablation(
        test_function=rosen,
        initial_point=( -1.5, 2.0 ),
        max_iterations=50,
        results_dir=results_dir,
        plots_dir=plots_dir
    )

    # Ensure summary file exists
    summary_csv = os.path.join(results_dir, 'optimizer_ablation_summary.csv')
    assert os.path.exists(summary_csv), "Summary CSV not written"

    df_summary = pd.read_csv(summary_csv)

    # Expected fair learning rates mapping
    expected_lr = {
        'SGD': 0.1,
        'SGD+Momentum': 0.1,
        'RMSProp': 0.001,
        'Adam': 0.001,
        'AdamW': 0.001,
        'AMSGrad': 0.001,
    }

    # Check that LR column exists and matches expected values
    assert 'LR' in df_summary.columns, 'LR column missing from summary'

    for idx, row in df_summary.iterrows():
        opt = str(row['Optimizer'])
        if opt in expected_lr:
            assert abs(float(row['LR']) - expected_lr[opt]) < 1e-12, f"LR for {opt} != expected fair LR"
