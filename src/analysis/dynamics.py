"""
Dynamics analysis utilities for optimizer trajectories.
"""
import numpy as np


def step_vectors_from_trajectory(trajectory):
    traj = np.asarray(trajectory)
    if traj.shape[0] < 2:
        return np.zeros((0, traj.shape[1]))
    return traj[1:] - traj[:-1]


def instantaneous_step_sizes(trajectory):
    steps = step_vectors_from_trajectory(trajectory)
    norms = np.linalg.norm(steps, axis=1)
    return norms


def smoothness_angles(trajectory):
    """Compute angle (radians) between consecutive update vectors. Returns array of length T-2."""
    steps = step_vectors_from_trajectory(trajectory)
    if steps.shape[0] < 2:
        return np.array([])
    dots = np.einsum('ij,ij->i', steps[1:], steps[:-1])
    norms = np.linalg.norm(steps[1:], axis=1) * np.linalg.norm(steps[:-1], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cosang = np.clip(dots / (norms + 1e-20), -1.0, 1.0)
        angles = np.arccos(cosang)
    # NaNs (from zero-length steps) -> large angle (pi/2)
    angles = np.nan_to_num(angles, nan=np.pi/2)
    return angles


def oscillation_flags(trajectory, threshold_radians=np.pi/2):
    """Return boolean array marking steps where turning angle > threshold (indicative of oscillation)."""
    angles = smoothness_angles(trajectory)
    return angles > threshold_radians


def add_dynamics_metrics(df, x_col='x', y_col='y'):
    """Given a pandas DataFrame with x,y columns ordered by iteration, add dynamics metrics as new columns.

    Adds: step_size (||theta_{t+1} - theta_t||), step_angle (angle between successive steps), oscillation_flag (bool)
    Returns new DataFrame with added columns (and summary stats in dict).
    """
    import pandas as pd
    traj = df[[x_col, y_col]].values
    step_sizes = instantaneous_step_sizes(traj)
    # Pad to match iterations: step_size for iteration i corresponds to step from i to i+1; set last to 0
    step_sizes_padded = np.concatenate([step_sizes, [0.0]])
    df = df.copy()
    df['step_size'] = step_sizes_padded

    angles = smoothness_angles(traj)
    angles_padded = np.concatenate([[np.nan], angles, [np.nan]])  # align so that angle at idx i is angle between steps i-1 and i
    df['step_angle'] = angles_padded
    df['step_angle_deg'] = np.degrees(df['step_angle'])

    osc_flags = oscillation_flags(traj)
    osc_padded = np.concatenate([[False], osc_flags, [False]])
    df['oscillation_flag'] = osc_padded

    # Summary
    summary = {
        'mean_step_size': float(np.nanmean(df['step_size'])),
        'median_step_size': float(np.nanmedian(df['step_size'])),
        'mean_angle_deg': float(np.nanmean(df['step_angle_deg'])),
        'oscillation_rate': float(np.mean(df['oscillation_flag']))
    }
    return df, summary
