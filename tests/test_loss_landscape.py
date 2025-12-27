import os
import numpy as np
from pathlib import Path


def test_plot_loss_landscape_saves(tmp_path):
    from src.visualization.loss_landscape import plot_loss_landscape

    def quad(pt):
        return float(pt[0]) ** 2 + float(pt[1]) ** 2

    out = tmp_path / "loss_landscape_test.png"
    res = plot_loss_landscape(quad, x_range=(-1, 1), y_range=(-1, 1), num_points=20, save_path=str(out))

    assert out.exists()
    assert str(out) == res


def test_create_loss_landscape_animation_saves(tmp_path):
    from src.visualization.loss_landscape import create_loss_landscape_animation

    def quad(pt):
        return float(pt[0]) ** 2 + float(pt[1]) ** 2

    out = tmp_path / "loss_landscape_anim.gif"
    res = create_loss_landscape_animation(quad, x_range=(-1, 1), y_range=(-1, 1), num_points=20, n_frames=5, save_path=str(out), interval=50)

    assert out.exists()
    assert str(out) == res
