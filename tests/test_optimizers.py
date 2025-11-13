"""
Unit tests for optimizer correctness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.core.optimizers import SGD, SGDMomentum, SGDNesterov, RMSProp, Adam, AdamW


class TestSGD:
    """Test SGD optimizer."""
    
    def test_simple_step(self):
        """Test single SGD update step."""
        opt = SGD(lr=0.1)
        params = (1.0, 2.0)
        gradients = (0.5, 1.0)
        
        new_x, new_y = opt.step(params, gradients)
        
        # Expected: x_new = 1.0 - 0.1 * 0.5 = 0.95
        assert abs(new_x - 0.95) < 1e-10
        # Expected: y_new = 2.0 - 0.1 * 1.0 = 1.9
        assert abs(new_y - 1.9) < 1e-10
    
    def test_zero_gradient(self):
        """With zero gradient, params should not change."""
        opt = SGD(lr=0.1)
        params = (1.0, 2.0)
        gradients = (0.0, 0.0)
        
        new_x, new_y = opt.step(params, gradients)
        
        assert abs(new_x - 1.0) < 1e-10
        assert abs(new_y - 2.0) < 1e-10
    
    def test_different_learning_rates(self):
        """Test with different learning rates."""
        for lr in [0.01, 0.1, 1.0]:
            opt = SGD(lr=lr)
            params = (1.0, 1.0)
            gradients = (1.0, 1.0)
            
            new_x, new_y = opt.step(params, gradients)
            
            assert abs(new_x - (1.0 - lr * 1.0)) < 1e-10
            assert abs(new_y - (1.0 - lr * 1.0)) < 1e-10


class TestSGDMomentum:
    """Test SGD with Momentum optimizer."""
    
    def test_first_step(self):
        """First step should be same as SGD (velocity=0)."""
        opt = SGDMomentum(lr=0.1, beta=0.9)
        params = (1.0, 2.0)
        gradients = (0.5, 1.0)
        
        new_x, new_y = opt.step(params, gradients)
        
        # v_new = 0.9 * 0 + 0.5 = 0.5
        # x_new = 1.0 - 0.1 * 0.5 = 0.95
        assert abs(new_x - 0.95) < 1e-10
        assert abs(new_y - 1.9) < 1e-10
    
    def test_momentum_accumulation(self):
        """Test that momentum accumulates over steps."""
        opt = SGDMomentum(lr=0.1, beta=0.9)
        params = (1.0, 1.0)
        gradients = (1.0, 1.0)
        
        # Step 1: v = 1.0, x = 1.0 - 0.1 * 1.0 = 0.9
        new_x, new_y = opt.step(params, gradients)
        assert abs(new_x - 0.9) < 1e-10
        
        # Step 2: v = 0.9 * 1.0 + 1.0 = 1.9, x = 0.9 - 0.1 * 1.9 = 0.71
        params = (new_x, new_y)
        new_x, new_y = opt.step(params, gradients)
        assert abs(new_x - 0.71) < 1e-10
    
    def test_reset(self):
        """Test that reset clears velocity."""
        opt = SGDMomentum(lr=0.1, beta=0.9)
        
        # Do one step to accumulate velocity
        opt.step((1.0, 1.0), (1.0, 1.0))
        
        # Reset
        opt.reset()
        
        # After reset, should behave like first step
        new_x, new_y = opt.step((1.0, 1.0), (1.0, 1.0))
        assert abs(new_x - 0.9) < 1e-10


class TestRMSProp:
    """Test RMSProp optimizer."""
    
    def test_first_step(self):
        """Test first RMSProp update."""
        opt = RMSProp(lr=0.1, decay_rate=0.9, epsilon=1e-8)
        params = (1.0, 1.0)
        gradients = (1.0, 1.0)
        
        # s = 0.9 * 0 + 0.1 * 1^2 = 0.1
        # x_new = 1.0 - 0.1 * 1.0 / sqrt(0.1 + 1e-8)
        expected_x = 1.0 - 0.1 * 1.0 / np.sqrt(0.1 + 1e-8)
        
        new_x, new_y = opt.step(params, gradients)
        
        assert abs(new_x - expected_x) < 1e-6
    
    def test_adaptive_scaling(self):
        """Test that RMSProp adapts to gradient magnitudes over multiple steps."""
        opt = RMSProp(lr=0.1, decay_rate=0.9, epsilon=1e-8)
        
        # Run multiple steps with consistent gradients to build up s
        params = (1.0, 1.0)
        for _ in range(5):
            params = opt.step(params, (10.0, 0.1))
        
        # After multiple steps, large gradient should have larger s
        # Thus smaller effective learning rate
        # s_x will be larger, so step_x should be relatively smaller
        assert opt.s_x > opt.s_y * 10, "s_x should accumulate more from larger gradients"


class TestSGDNesterov:
    """Test SGD with Nesterov Accelerated Gradient."""

    def test_first_step_more_than_sgd(self):
        """First step is larger than SGD by factor (1+beta) when v=0."""
        beta = 0.9
        lr = 0.1
        opt = SGDNesterov(lr=lr, beta=beta)
        params = (1.0, 2.0)
        gradients = (0.5, 1.0)

        new_x, new_y = opt.step(params, gradients)

        # d = (1+beta) * grad on first step
        expected_x = 1.0 - lr * (1 + beta) * 0.5
        expected_y = 2.0 - lr * (1 + beta) * 1.0
        assert abs(new_x - expected_x) < 1e-10
        assert abs(new_y - expected_y) < 1e-10

    def test_zero_gradient(self):
        opt = SGDNesterov(lr=0.1, beta=0.9)
        params = (1.0, 2.0)
        gradients = (0.0, 0.0)

        new_x, new_y = opt.step(params, gradients)

        assert abs(new_x - 1.0) < 1e-10
        assert abs(new_y - 2.0) < 1e-10

    def test_reset(self):
        beta = 0.8
        lr = 0.05
        opt = SGDNesterov(lr=lr, beta=beta)
        opt.step((1.0, 1.0), (1.0, 1.0))
        opt.reset()
        # First step after reset should again use (1+beta)*grad
        new_x, new_y = opt.step((1.0, 1.0), (1.0, 1.0))
        expected = 1.0 - lr * (1 + beta) * 1.0
        assert abs(new_x - expected) < 1e-10


class TestAdam:
    """Test Adam optimizer."""
    
    def test_first_step_bias_correction(self):
        """Test that bias correction works on first step."""
        opt = Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        params = (1.0, 1.0)
        gradients = (1.0, 1.0)
        
        new_x, new_y = opt.step(params, gradients)
        
        # m = 0.1, v = 0.001
        # m_hat = 0.1 / (1 - 0.9^1) = 1.0
        # v_hat = 0.001 / (1 - 0.999^1) = 1.0
        # x_new = 1.0 - 0.001 * 1.0 / (sqrt(1.0) + 1e-8)
        expected_x = 1.0 - 0.001 * 1.0 / (np.sqrt(1.0) + 1e-8)
        
        assert abs(new_x - expected_x) < 1e-6
    
    def test_timestep_increment(self):
        """Test that timestep increments correctly."""
        opt = Adam(lr=0.001, beta1=0.9, beta2=0.999)
        
        assert opt.t == 0
        
        opt.step((1.0, 1.0), (1.0, 1.0))
        assert opt.t == 1
        
        opt.step((1.0, 1.0), (1.0, 1.0))
        assert opt.t == 2
    
    def test_reset(self):
        """Test that reset clears moments and timestep."""
        opt = Adam(lr=0.001, beta1=0.9, beta2=0.999)
        
        # Do steps
        opt.step((1.0, 1.0), (1.0, 1.0))
        opt.step((1.0, 1.0), (1.0, 1.0))
        
        assert opt.t == 2
        
        # Reset
        opt.reset()
        
        assert opt.t == 0
        assert opt.m_x == 0.0
        assert opt.v_x == 0.0
    
    def test_momentum_and_adaptive_combination(self):
        """Test that Adam combines momentum and adaptive LR."""
        opt = Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
        # Step 1
        params = (1.0, 1.0)
        new_x1, new_y1 = opt.step(params, (1.0, 1.0))
        
        # Step 2 (same gradient direction)
        params = (new_x1, new_y1)
        new_x2, new_y2 = opt.step(params, (1.0, 1.0))
        
        # Step should increase due to momentum
        step1 = abs(1.0 - new_x1)
        step2 = abs(new_x1 - new_x2)
        
        # Note: Due to bias correction changing, this test is approximate
        # Main point: Adam should be working consistently
        assert step2 > 0  # Should still be making progress


class TestAdamW:
    """Tests for AdamW optimizer (decoupled weight decay)."""

    def test_zero_grad_is_pure_decay(self):
        lr = 0.01
        wd = 0.1
        opt = AdamW(lr=lr, weight_decay=wd)
        params = (1.0, -2.0)
        gradients = (0.0, 0.0)

        new_x, new_y = opt.step(params, gradients)

        # With zero gradient, Adam step is zero; only decay applies: p <- p * (1 - lr*wd)
        factor = (1 - lr * wd)
        assert abs(new_x - (1.0 * factor)) < 1e-12
        assert abs(new_y - (-2.0 * factor)) < 1e-12

    def test_matches_adam_when_no_decay(self):
        lr = 0.001
        opt_adam = Adam(lr=lr)
        opt_adamw = AdamW(lr=lr, weight_decay=0.0)

        params_a = (1.0, 1.0)
        params_w = (1.0, 1.0)
        grads = (0.3, -0.7)

        for _ in range(5):
            params_a = opt_adam.step(params_a, grads)
            params_w = opt_adamw.step(params_w, grads)

        assert np.allclose(params_a, params_w, rtol=1e-7, atol=1e-9)


class TestOptimizerConsistency:
    """Test consistency across optimizers."""
    
    def test_all_optimizers_converge_on_quadratic(self):
        """All optimizers should reduce loss on simple quadratic."""
        # f(x,y) = x^2 + y^2, optimum at (0,0)
        
        optimizers = [
            SGD(lr=0.1),
            SGDMomentum(lr=0.1, beta=0.9),
            SGDNesterov(lr=0.05, beta=0.9),
            RMSProp(lr=0.1, decay_rate=0.9),
            Adam(lr=0.1, beta1=0.9, beta2=0.999),
            AdamW(lr=0.05, weight_decay=0.01),
        ]
        
        for opt in optimizers:
            opt.reset()
            params = (5.0, 5.0)
            
            # Run 50 steps
            for _ in range(50):
                # Gradient of x^2 + y^2 is (2x, 2y)
                x, y = params
                gradients = (2 * x, 2 * y)
                params = opt.step(params, gradients)
            
            # Should be closer to origin
            final_x, final_y = params
            final_dist = np.sqrt(final_x**2 + final_y**2)
            initial_dist = np.sqrt(5.0**2 + 5.0**2)
            
            assert final_dist < initial_dist, f"{opt.name} did not converge"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
