import numpy as np
import pytest

from src.core.test_functions import Ackley2D


def numerical_grad(f, x, y, eps=1e-6):
    fx1 = f(x + eps, y)
    fx2 = f(x - eps, y)
    fy1 = f(x, y + eps)
    fy2 = f(x, y - eps)
    return (fx1 - fx2) / (2 * eps), (fy1 - fy2) / (2 * eps)


def test_ackley2d_gradient_matches_numerical():
    ack = Ackley2D()
    # Test a few non-singular points
    points = [(-1.2, 0.7), (0.3, -0.8), (2.0, 1.5), (-3.1, 0.4), (0.5, 0.5)]
    tol = 1e-4
    for x, y in points:
        gx, gy = ack.gradient(x, y)
        ngx, ngy = numerical_grad(ack.compute, x, y)
        assert np.isfinite(gx) and np.isfinite(gy)
        assert np.allclose(gx, ngx, atol=tol, rtol=1e-3)
        assert np.allclose(gy, ngy, atol=tol, rtol=1e-3)
