import os
import pandas as pd
from src.experiments.run_optimizer_ablation import run_optimizer_ablation
from src.core.test_functions import Rosenbrock


def test_theory_overlay_and_dynamics_written(tmp_path):
    results_dir = str(tmp_path / "results")
    plots_dir = str(tmp_path / "plots")

    rosen = Rosenbrock(a=1, b=100)
    df = run_optimizer_ablation(
        test_function=rosen,
        initial_point=( -1.5, 2.0 ),
        max_iterations=40,
        results_dir=results_dir,
        plots_dir=plots_dir
    )

    # Ensure summary CSV exists and Has_Theoretical_Curve column present
    summary_csv = os.path.join(results_dir, 'optimizer_ablation_summary.csv')
    assert os.path.exists(summary_csv), "Summary CSV not written"

    df_summary = pd.read_csv(summary_csv)
    assert 'Has_Theoretical_Curve' in df_summary.columns, 'Has_Theoretical_Curve missing'

    # At least one optimizer should have a theoretical curve (heuristic possibility)
    assert df_summary['Has_Theoretical_Curve'].any(), 'No theoretical curves were recorded'

    # Check dynamics CSV files exist for at least one optimizer
    dyn_dir = os.path.join(plots_dir, 'dynamics')
    assert os.path.exists(dyn_dir), 'Dynamics directory missing'

    has_csv = False
    for opt_name in df_summary['Optimizer']:
        safe_name = opt_name.replace(' ', '_')
        candidate = os.path.join(dyn_dir, safe_name, f"{opt_name}_dynamics.csv")
        if os.path.exists(candidate):
            has_csv = True
            # Verify it contains expected columns
            dd = pd.read_csv(candidate)
            assert 'loss' in dd.columns and 'grad_norm' in dd.columns
            break
    assert has_csv, 'No per-optimizer dynamics CSV found'
