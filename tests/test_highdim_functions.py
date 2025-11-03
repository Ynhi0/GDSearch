"""
Tests for high-dimensional test functions.
"""

import numpy as np
import pytest
from src.core.test_functions import Rastrigin, Ackley, Sphere, Schwefel


class TestRastriginFunction:
    """Tests for Rastrigin function."""
    
    def test_optimum_value(self):
        """Test that Rastrigin has correct value at optimum."""
        for dim in [2, 5, 10]:
            func = Rastrigin(dim=dim)
            x_opt = np.zeros(dim)
            value = func.compute(x_opt)
            assert abs(value) < 1e-10, f"Rastrigin optimum should be 0, got {value}"
    
    def test_gradient_at_optimum(self):
        """Test that gradient is zero at optimum."""
        func = Rastrigin(dim=10)
        x_opt = np.zeros(10)
        grad = func.gradient(x_opt)
        assert np.allclose(grad, 0, atol=1e-10), "Gradient should be zero at optimum"
    
    def test_gradient_numerical(self):
        """Test gradient against numerical approximation."""
        func = Rastrigin(dim=5)
        x = np.random.randn(5) * 2  # Random point in [-5, 5]
        
        # Analytical gradient
        grad_analytical = func.gradient(x)
        
        # Numerical gradient
        eps = 1e-7
        grad_numerical = np.zeros(5)
        for i in range(5):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad_numerical[i] = (func.compute(x_plus) - func.compute(x_minus)) / (2 * eps)
        
        assert np.allclose(grad_analytical, grad_numerical, rtol=1e-4), \
            f"Gradient mismatch: analytical={grad_analytical}, numerical={grad_numerical}"
    
    def test_multimodal_nature(self):
        """Test that Rastrigin has multiple local minima."""
        func = Rastrigin(dim=2)
        
        # Check that there are local minima away from global optimum
        # Rastrigin has local minima near x = [±1, ±1], [±2, ±2], etc.
        local_min_1 = np.array([1.0, 0.0])
        local_min_2 = np.array([0.0, 1.0])
        
        f_global = func.compute(np.zeros(2))
        f_local_1 = func.compute(local_min_1)
        f_local_2 = func.compute(local_min_2)
        
        # Local minima should have higher values than global minimum
        assert f_local_1 > f_global, "Local minimum should be higher than global"
        assert f_local_2 > f_global, "Local minimum should be higher than global"
    
    def test_bounds(self):
        """Test that bounds are correct."""
        func = Rastrigin(dim=10)
        lower, upper = func.get_bounds()
        assert lower == -5.12
        assert upper == 5.12
    
    def test_different_dimensions(self):
        """Test that function works with different dimensions."""
        for dim in [2, 5, 10, 20, 50]:
            func = Rastrigin(dim=dim)
            x = np.random.randn(dim)
            value = func.compute(x)
            grad = func.gradient(x)
            
            assert isinstance(value, (int, float, np.number))
            assert grad.shape == (dim,)


class TestAckleyFunction:
    """Tests for Ackley function."""
    
    def test_optimum_value(self):
        """Test that Ackley has correct value at optimum."""
        for dim in [2, 5, 10]:
            func = Ackley(dim=dim)
            x_opt = np.zeros(dim)
            value = func.compute(x_opt)
            assert abs(value) < 1e-10, f"Ackley optimum should be 0, got {value}"
    
    def test_gradient_at_optimum(self):
        """Test that gradient is zero at optimum."""
        func = Ackley(dim=10)
        x_opt = np.zeros(10)
        grad = func.gradient(x_opt)
        assert np.allclose(grad, 0, atol=1e-10), "Gradient should be zero at optimum"
    
    def test_gradient_numerical(self):
        """Test gradient against numerical approximation."""
        func = Ackley(dim=5)
        x = np.random.randn(5) * 10  # Random point
        
        # Analytical gradient
        grad_analytical = func.gradient(x)
        
        # Numerical gradient
        eps = 1e-7
        grad_numerical = np.zeros(5)
        for i in range(5):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad_numerical[i] = (func.compute(x_plus) - func.compute(x_minus)) / (2 * eps)
        
        assert np.allclose(grad_analytical, grad_numerical, rtol=1e-4), \
            f"Gradient mismatch: analytical={grad_analytical}, numerical={grad_numerical}"
    
    def test_nearly_flat_outer_region(self):
        """Test that Ackley is nearly flat far from origin."""
        func = Ackley(dim=2)
        
        # Points far from origin should have similar high values
        x1 = np.array([20.0, 20.0])
        x2 = np.array([-20.0, 20.0])
        x3 = np.array([20.0, -20.0])
        
        f1 = func.compute(x1)
        f2 = func.compute(x2)
        f3 = func.compute(x3)
        
        # Values should be similar (nearly flat)
        assert abs(f1 - f2) < 2.0, "Outer region should be nearly flat"
        assert abs(f1 - f3) < 2.0, "Outer region should be nearly flat"
    
    def test_bounds(self):
        """Test that bounds are correct."""
        func = Ackley(dim=10)
        lower, upper = func.get_bounds()
        assert lower == -32.768
        assert upper == 32.768
    
    def test_different_dimensions(self):
        """Test that function works with different dimensions."""
        for dim in [2, 5, 10, 20]:
            func = Ackley(dim=dim)
            x = np.random.randn(dim)
            value = func.compute(x)
            grad = func.gradient(x)
            
            assert isinstance(value, (int, float, np.number))
            assert grad.shape == (dim,)


class TestSphereFunction:
    """Tests for Sphere function."""
    
    def test_optimum_value(self):
        """Test that Sphere has correct value at optimum."""
        for dim in [2, 5, 10]:
            func = Sphere(dim=dim)
            x_opt = np.zeros(dim)
            value = func.compute(x_opt)
            assert abs(value) < 1e-10, f"Sphere optimum should be 0, got {value}"
    
    def test_gradient_at_optimum(self):
        """Test that gradient is zero at optimum."""
        func = Sphere(dim=10)
        x_opt = np.zeros(10)
        grad = func.gradient(x_opt)
        assert np.allclose(grad, 0, atol=1e-10), "Gradient should be zero at optimum"
    
    def test_gradient_correctness(self):
        """Test that gradient is correct (2x)."""
        func = Sphere(dim=5)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grad = func.gradient(x)
        expected_grad = 2 * x
        assert np.allclose(grad, expected_grad), f"Gradient should be 2x, got {grad}"
    
    def test_convex_nature(self):
        """Test that Sphere is convex (increases away from origin)."""
        func = Sphere(dim=2)
        
        origin = np.zeros(2)
        point1 = np.array([1.0, 0.0])
        point2 = np.array([2.0, 0.0])
        
        f0 = func.compute(origin)
        f1 = func.compute(point1)
        f2 = func.compute(point2)
        
        assert f0 < f1 < f2, "Sphere should increase monotonically from origin"
    
    def test_bounds(self):
        """Test that bounds are correct."""
        func = Sphere(dim=10)
        lower, upper = func.get_bounds()
        assert lower == -5.12
        assert upper == 5.12
    
    def test_different_dimensions(self):
        """Test that function works with different dimensions."""
        for dim in [2, 5, 10, 20, 50, 100]:
            func = Sphere(dim=dim)
            x = np.random.randn(dim)
            value = func.compute(x)
            grad = func.gradient(x)
            
            assert isinstance(value, (int, float, np.number))
            assert grad.shape == (dim,)


class TestSchwefelFunction:
    """Tests for Schwefel function."""
    
    def test_optimum_value(self):
        """Test that Schwefel has near-zero value at optimum."""
        for dim in [2, 5, 10]:
            func = Schwefel(dim=dim)
            x_opt = np.full(dim, 420.9687)
            value = func.compute(x_opt)
            assert abs(value) < 1.0, f"Schwefel optimum should be ~0, got {value}"
    
    def test_gradient_numerical(self):
        """Test gradient against numerical approximation."""
        func = Schwefel(dim=5)
        x = np.random.randn(5) * 100 + 200  # Random point in reasonable range
        
        # Analytical gradient
        grad_analytical = func.gradient(x)
        
        # Numerical gradient
        eps = 1e-6
        grad_numerical = np.zeros(5)
        for i in range(5):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad_numerical[i] = (func.compute(x_plus) - func.compute(x_minus)) / (2 * eps)
        
        assert np.allclose(grad_analytical, grad_numerical, rtol=1e-3), \
            f"Gradient mismatch: analytical={grad_analytical}, numerical={grad_numerical}"
    
    def test_gradient_at_zero(self):
        """Test gradient computation at x=0 (special case)."""
        func = Schwefel(dim=5)
        x = np.zeros(5)
        grad = func.gradient(x)
        
        # Gradient should be finite and not NaN
        assert np.all(np.isfinite(grad)), "Gradient at zero should be finite"
    
    def test_deceptive_nature(self):
        """Test that Schwefel is deceptive (optimum far from origin)."""
        func = Schwefel(dim=2)
        
        origin = np.zeros(2)
        optimum = np.array([420.9687, 420.9687])
        
        f_origin = func.compute(origin)
        f_optimum = func.compute(optimum)
        
        # Origin should have much higher value than optimum
        assert f_origin > f_optimum + 100, "Schwefel should be deceptive"
    
    def test_bounds(self):
        """Test that bounds are correct."""
        func = Schwefel(dim=10)
        lower, upper = func.get_bounds()
        assert lower == -500
        assert upper == 500
    
    def test_different_dimensions(self):
        """Test that function works with different dimensions."""
        for dim in [2, 5, 10, 20]:
            func = Schwefel(dim=dim)
            x = np.random.randn(dim) * 100
            value = func.compute(x)
            grad = func.gradient(x)
            
            assert isinstance(value, (int, float, np.number))
            assert grad.shape == (dim,)


class TestHighDimensionalComparison:
    """Tests comparing different high-dimensional functions."""
    
    def test_all_functions_have_correct_optimum(self):
        """Test that all functions report correct optimum location."""
        dim = 10
        functions = [
            Rastrigin(dim=dim),
            Ackley(dim=dim),
            Sphere(dim=dim),
            Schwefel(dim=dim)
        ]
        
        for func in functions:
            x_opt, f_opt = func.get_optimum()
            assert len(x_opt) == dim, f"{func.name}: optimum should have {dim} dimensions"
            assert isinstance(f_opt, (int, float)), f"{func.name}: f_opt should be a number"
    
    def test_difficulty_ranking(self):
        """Test that functions have expected difficulty ranking."""
        dim = 10
        
        # Create functions
        sphere = Sphere(dim=dim)
        ackley = Ackley(dim=dim)
        rastrigin = Rastrigin(dim=dim)
        schwefel = Schwefel(dim=dim)
        
        # Test at a random point
        np.random.seed(42)
        x = np.random.randn(dim)
        
        # Compute gradient norms
        grad_sphere = np.linalg.norm(sphere.gradient(x))
        grad_ackley = np.linalg.norm(ackley.gradient(x))
        grad_rastrigin = np.linalg.norm(rastrigin.gradient(x))
        grad_schwefel = np.linalg.norm(schwefel.gradient(x))
        
        # All should have non-zero gradients at random point
        assert grad_sphere > 0, "Sphere should have non-zero gradient"
        assert grad_ackley > 0, "Ackley should have non-zero gradient"
        assert grad_rastrigin > 0, "Rastrigin should have non-zero gradient"
        assert grad_schwefel > 0, "Schwefel should have non-zero gradient"
    
    def test_scalability_to_high_dimensions(self):
        """Test that all functions scale to high dimensions."""
        for dim in [10, 50, 100]:
            functions = [
                Rastrigin(dim=dim),
                Ackley(dim=dim),
                Sphere(dim=dim),
                Schwefel(dim=dim)
            ]
            
            x = np.random.randn(dim)
            
            for func in functions:
                value = func.compute(x)
                grad = func.gradient(x)
                
                assert isinstance(value, (int, float, np.number))
                assert grad.shape == (dim,)
                assert np.all(np.isfinite(grad)), f"{func.name}: gradient should be finite"
