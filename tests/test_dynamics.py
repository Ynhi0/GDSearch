import numpy as np
from src.analysis.dynamics import instantaneous_step_sizes, smoothness_angles, add_dynamics_metrics


def test_step_sizes_and_angles():
    traj = np.array([[0.0,0.0],[1.0,0.0],[2.0,0.0],[2.0,1.0]])
    sizes = instantaneous_step_sizes(traj)
    assert np.allclose(sizes, [1.0, 1.0, 1.0])
    angles = smoothness_angles(traj)
    # First two steps are colinear -> small angle ~ 0, second angle between [1,0] and [0,1] -> 90deg
    assert angles.shape[0] == 2
    # angle[0] near 0
    assert angles[0] < 1e-6
    assert np.isclose(np.degrees(angles[1]), 90.0)


def test_add_dynamics_metrics_adds_columns():
    import pandas as pd
    df = pd.DataFrame({'iteration':[0,1,2],'x':[0.0,1.0,2.0],'y':[0.0,0.0,0.0]})
    df2, summary = add_dynamics_metrics(df)
    assert 'step_size' in df2.columns
    assert 'step_angle' in df2.columns
    assert 'oscillation_flag' in df2.columns
    assert 'mean_step_size' in summary
