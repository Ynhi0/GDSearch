import tempfile
from pathlib import Path

from src.visualization.trajectory_2d import (
    compare_momentum_beta_trajectories,
    compare_adam_beta_trajectories,
    compare_optimizer_families,
)


def test_visualization_accepts_output_dir(tmp_path):
    out = str(tmp_path)

    # Should not raise
    compare_momentum_beta_trajectories(test_function='rosenbrock', beta_values=[0.0, 0.9], output_dir=out)
    compare_adam_beta_trajectories(test_function='rosenbrock', beta_configs=[(0.9, 0.999)], output_dir=out)
    compare_optimizer_families(test_function='rosenbrock', output_dir=out)

    # Check that files were created
    files = list(Path(out).glob('*.png'))
    assert len(files) > 0, "Expected generated PNG plots in the provided output dir"
