"""
Demo script: Saddle point escape visualization

Generates:
 - Vector field (quiver) for SaddlePoint
 - Trajectories comparison starting close to the saddle for SGD, Momentum, Adam

Usage: python scripts/demo_saddle_point.py
"""
from pathlib import Path
import numpy as np
import logging

from src.visualization.trajectory_2d import (
    plot_vector_field_overlay,
    compare_optimizer_families,
)
from src.core.test_functions import SaddlePoint


def run_demo(save_dir: str = 'results/demo_saddle'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    # Vector field for saddle
    test_fn = SaddlePoint()
    func = test_fn.compute
    grad_func = lambda x, y: test_fn.gradient(x, y)
    xlim, ylim = (-2, 2), (-2, 2)

    vf_path = Path(save_dir) / 'saddle_vector_field.png'
    plot_vector_field_overlay(func, grad_func, xlim, ylim, vf_path, density=30, normalize=True)

    # Trajectory comparison starting extremely close to the saddle (but not exactly at it)
    print("Generating trajectories starting near saddle (close to origin)...")
    # compare_optimizer_families has an option for 'saddle' when invoked with that keyword
    compare_optimizer_families(test_function='saddle', save_dir=save_dir)

    print(f"Demo outputs saved to: {save_dir}")


if __name__ == '__main__':
    run_demo()
