"""
Integration tests for quick end-to-end pipeline checks.
These tests are intentionally lightweight (quick mode / short runs) so they can run in CI.
"""

import tempfile
from pathlib import Path
import os

from src.experiments.convergence_rate_validation import validate_convergence_rate
from src.experiments.cross_optimizer_dynamics_comparison import run_cross_optimizer_dynamics_comparison


def test_validate_convergence_rate_quick():
    # Run a very short convergence validation on ill_conditioned quadratics
    res = validate_convergence_rate(
        optimizer_name='SGD_test',
        optimizer_class=lambda **kw: __import__('types').SimpleNamespace(step=lambda params, grad: params - 0.01 * grad),
        optimizer_params={},
        test_function='ill_conditioned',
        x0=[-1.0, 1.0],
        max_iters=20,
        tol=1e-6,
        noise_std=0.0
    )

    assert 'grad_norms' in res
    assert len(res['grad_norms']) <= 20
    assert 'fit_results' in res


def test_cross_optimizer_dynamics_quick(tmp_path: Path):
    out_dir = str(tmp_path / "dynamics_quick")
    # Run with a single optimizer, single seed, and 1 epoch to keep it quick
    df = run_cross_optimizer_dynamics_comparison(
        dataset='MNIST',
        optimizers=['SGD'],
        epochs=1,
        seeds=[42],
        quick=True,
        results_dir=out_dir
    )

    # Expect resulting CSV to be written
    csv_path = Path(out_dir) / "cross_optimizer_dynamics_MNIST.csv"
    assert csv_path.exists(), f"Expected dynamics CSV at {csv_path}"
    # DataFrame returned should have at least one row
    assert not df.empty
