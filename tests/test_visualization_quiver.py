import os
from pathlib import Path

import numpy as np

from src.visualization.trajectory_2d import plot_vector_field_overlay
from src.core.test_functions import SaddlePoint


def test_plot_vector_field_creates_file(tmp_path):
    save_dir = tmp_path
    out_file = save_dir / 'vf_test.png'

    fn = SaddlePoint()
    func = fn.compute
    grad = lambda x, y: np.array(fn.gradient(x, y))

    # Run with low density to keep the test fast
    plot_vector_field_overlay(func, grad, (-1, 1), (-1, 1), out_file, density=8, normalize=True)

    assert out_file.exists()
    stat = out_file.stat()
    assert stat.st_size > 0
