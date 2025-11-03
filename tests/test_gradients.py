"""
Unit tests for gradient correctness using numerical verification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.core.test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint


def numerical_gradient(func, x, y, eps=1e-7):
    """
    Compute numerical gradient using central differences.
    
    Args:
        func: Test function object
        x, y: Point to evaluate gradient
        eps: Step size for finite differences
        
    Returns:
        (grad_x, grad_y): Numerical gradient
    """
    grad_x = (func.compute(x + eps, y) - func.compute(x - eps, y)) / (2 * eps)
    grad_y = (func.compute(x, y + eps) - func.compute(x, y - eps)) / (2 * eps)
    return grad_x, grad_y


def numerical_hessian(func, x, y, eps=1e-5):
    """
    Compute numerical Hessian using finite differences.
    
    Returns:
        2x2 numpy array
    """
    # Second derivatives
    h_xx = (func.compute(x + eps, y) - 2 * func.compute(x, y) + func.compute(x - eps, y)) / (eps ** 2)
    h_yy = (func.compute(x, y + eps) - 2 * func.compute(x, y) + func.compute(x, y - eps)) / (eps ** 2)
    
    # Mixed derivative
    h_xy = (func.compute(x + eps, y + eps) - func.compute(x + eps, y - eps) - 
            func.compute(x - eps, y + eps) + func.compute(x - eps, y - eps)) / (4 * eps ** 2)
    
    return np.array([[h_xx, h_xy], [h_xy, h_yy]])


class TestRosenbrockGradients:
    """Test Rosenbrock function gradients and Hessian."""
    
    @pytest.fixture
    def func(self):
        return Rosenbrock(a=1, b=100)
    
    @pytest.mark.parametrize("x,y", [
        (0.5, 0.5),
        (1.0, 1.0),
        (-1.0, 2.0),
        (1.5, 2.5),
        (0.0, 0.0),
    ])
    def test_gradient_correctness(self, func, x, y):
        """Verify analytic gradient matches numerical gradient."""
        # Analytic gradient
        grad_x_analytic, grad_y_analytic = func.gradient(x, y)
        
        # Numerical gradient
        grad_x_numerical, grad_y_numerical = numerical_gradient(func, x, y)
        
        # Check with tight tolerance
        assert abs(grad_x_analytic - grad_x_numerical) < 1e-5, \
            f"Gradient X mismatch at ({x}, {y}): {grad_x_analytic} vs {grad_x_numerical}"
        assert abs(grad_y_analytic - grad_y_numerical) < 1e-5, \
            f"Gradient Y mismatch at ({x}, {y}): {grad_y_analytic} vs {grad_y_numerical}"
    
    @pytest.mark.parametrize("x,y", [
        (0.5, 0.5),
        (1.0, 1.0),
        (-1.0, 2.0),
    ])
    def test_hessian_correctness(self, func, x, y):
        """Verify analytic Hessian matches numerical Hessian."""
        # Analytic Hessian
        hessian_analytic = func.hessian(x, y)
        
        # Numerical Hessian
        hessian_numerical = numerical_hessian(func, x, y)
        
        # Check all elements
        assert np.allclose(hessian_analytic, hessian_numerical, atol=1e-3), \
            f"Hessian mismatch at ({x}, {y})"
    
    def test_gradient_at_optimum(self, func):
        """At optimum (1, 1), gradient should be ~0."""
        grad_x, grad_y = func.gradient(1.0, 1.0)
        assert abs(grad_x) < 1e-10, f"Gradient X at optimum should be 0, got {grad_x}"
        assert abs(grad_y) < 1e-10, f"Gradient Y at optimum should be 0, got {grad_y}"


class TestIllConditionedQuadraticGradients:
    """Test IllConditionedQuadratic function gradients and Hessian."""
    
    @pytest.fixture
    def func(self):
        return IllConditionedQuadratic(kappa=100)
    
    @pytest.mark.parametrize("x,y", [
        (0.5, 0.5),
        (1.0, 1.0),
        (-1.0, 2.0),
        (2.0, -2.0),
    ])
    def test_gradient_correctness(self, func, x, y):
        """Verify analytic gradient matches numerical gradient."""
        grad_x_analytic, grad_y_analytic = func.gradient(x, y)
        grad_x_numerical, grad_y_numerical = numerical_gradient(func, x, y)
        
        assert abs(grad_x_analytic - grad_x_numerical) < 1e-5
        assert abs(grad_y_analytic - grad_y_numerical) < 1e-5
    
    @pytest.mark.parametrize("x,y", [
        (0.5, 0.5),
        (1.0, 1.0),
    ])
    def test_hessian_correctness(self, func, x, y):
        """Verify Hessian is diagonal with correct values."""
        hessian = func.hessian(x, y)
        
        # Should be diagonal
        assert abs(hessian[0, 1]) < 1e-10, "Off-diagonal should be 0"
        assert abs(hessian[1, 0]) < 1e-10, "Off-diagonal should be 0"
        
        # Diagonal values
        assert abs(hessian[0, 0] - 100) < 1e-10, "H[0,0] should be kappa=100"
        assert abs(hessian[1, 1] - 1) < 1e-10, "H[1,1] should be 1"
    
    def test_gradient_at_optimum(self, func):
        """At optimum (0, 0), gradient should be 0."""
        grad_x, grad_y = func.gradient(0.0, 0.0)
        assert abs(grad_x) < 1e-10
        assert abs(grad_y) < 1e-10


class TestSaddlePointGradients:
    """Test SaddlePoint function gradients and Hessian."""
    
    @pytest.fixture
    def func(self):
        return SaddlePoint()
    
    @pytest.mark.parametrize("x,y", [
        (0.5, 0.5),
        (1.0, 1.0),
        (-1.0, 2.0),
        (0.0, 0.0),
    ])
    def test_gradient_correctness(self, func, x, y):
        """Verify analytic gradient matches numerical gradient."""
        grad_x_analytic, grad_y_analytic = func.gradient(x, y)
        grad_x_numerical, grad_y_numerical = numerical_gradient(func, x, y)
        
        assert abs(grad_x_analytic - grad_x_numerical) < 1e-5
        assert abs(grad_y_analytic - grad_y_numerical) < 1e-5
    
    def test_hessian_correctness(self, func):
        """Verify Hessian has eigenvalues [1, -1] (saddle point)."""
        hessian = func.hessian(0.0, 0.0)
        
        # Should be diagonal [1, -1]
        assert abs(hessian[0, 0] - 1) < 1e-10
        assert abs(hessian[1, 1] - (-1)) < 1e-10
        assert abs(hessian[0, 1]) < 1e-10
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(hessian)
        assert abs(eigenvalues[0] - (-1)) < 1e-10, "Should have negative eigenvalue"
        assert abs(eigenvalues[1] - 1) < 1e-10, "Should have positive eigenvalue"
    
    def test_gradient_at_saddle(self, func):
        """At saddle point (0, 0), gradient should be 0."""
        grad_x, grad_y = func.gradient(0.0, 0.0)
        assert abs(grad_x) < 1e-10
        assert abs(grad_y) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
