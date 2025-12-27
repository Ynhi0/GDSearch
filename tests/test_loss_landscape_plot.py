import numpy as np
from src.visualization.loss_landscape import plot_loss_landscape, create_loss_landscape_animation


def test_plot_loss_landscape_runs(tmp_path):
    def sq(x):
        return x[0]**2 + x[1]**2
    out = tmp_path / "landscape.png"
    path = plot_loss_landscape(sq, x_range=(-1,1), y_range=(-1,1), num_points=30, save_path=str(out))
    assert str(out) == path


def test_create_animation_runs(tmp_path):
    def sq(x):
        return x[0]**2 + x[1]**2
    traj = np.array([[0.0,0.0],[0.5,0.5],[0.1,0.2],[0.0,0.0]])
    out = tmp_path / "traj.gif"
    path = create_loss_landscape_animation(sq, traj, x_range=(-1,1), y_range=(-1,1), num_points=30, save_path=str(out), fps=2)
    assert str(out) in path
