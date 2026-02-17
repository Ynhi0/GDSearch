"""
Comprehensive test suite for optimizer state persistence (state_dict/load_state_dict).

This addresses BUG #9 from the deep audit report - previously there were ZERO tests
verifying that optimizers can correctly save and restore their internal state.

Critical for:
- Checkpoint/resume functionality
- Multi-seed experiment reproducibility
- Long-running training reliability
"""

import numpy as np
import pytest
from src.core.optimizers import (
    SGD, SGDMomentum, SGDNesterov, RMSProp,
    Adam, AdamW, AMSGrad, RAdam, LAMB, AdaBound,
    SAM, Lookahead
)


class TestOptimizerStatePersistence:
    """Test that all optimizers correctly implement state_dict/load_state_dict."""
    
    def test_adam_state_persistence_tuple(self):
        """Test Adam state can be saved and restored (2D tuple params)."""
        opt1 = Adam(lr=0.001, beta1=0.9, beta2=0.999)
        
        # Run 10 steps to build up state
        params = (1.0, 2.0)
        for i in range(10):
            grads = (0.1 * (i + 1), 0.2 * (i + 1))
            params = opt1.step(params, grads)
        
        # Save state
        state = opt1.state_dict()
        
        # Verify state contains all required fields
        assert 'm_x' in state
        assert 'm_y' in state
        assert 'v_x' in state
        assert 'v_y' in state
        assert 't' in state
        assert state['t'] == 10
        
        # Create new optimizer and restore
        opt2 = Adam(lr=0.001, beta1=0.9, beta2=0.999)
        opt2.load_state_dict(state)
        
        # Verify states match
        assert opt2.m_x == opt1.m_x
        assert opt2.m_y == opt1.m_y
        assert opt2.v_x == opt1.v_x
        assert opt2.v_y == opt1.v_y
        assert opt2.t == opt1.t
        
        # Run one more step with both - should produce IDENTICAL results
        test_grads = (0.15, 0.25)
        params1 = opt1.step(params, test_grads)
        params2 = opt2.step(params, test_grads)
        
        assert abs(params1[0] - params2[0]) < 1e-10
        assert abs(params1[1] - params2[1]) < 1e-10
    
    def test_adam_state_persistence_array(self):
        """Test Adam state can be saved and restored (array params)."""
        opt1 = Adam(lr=0.001)
        
        # Run steps with array parameters
        params = np.random.randn(50)
        for _ in range(10):
            grads = np.random.randn(50) * 0.1
            params = opt1.step(params, grads)
        
        # Save and restore
        state = opt1.state_dict()
        opt2 = Adam(lr=0.001)
        opt2.load_state_dict(state)
        
        # Verify array states match
        assert opt2.t == opt1.t
        np.testing.assert_array_almost_equal(opt2.m, opt1.m)
        np.testing.assert_array_almost_equal(opt2.v, opt1.v)
        
        # Verify identical updates
        test_grads = np.random.randn(50) * 0.1
        params1 = opt1.step(params, test_grads)
        params2 = opt2.step(params, test_grads)
        np.testing.assert_array_almost_equal(params1, params2)
    
    def test_adamw_state_persistence(self):
        """Test AdamW state persistence (same structure as Adam)."""
        opt1 = AdamW(lr=0.001, weight_decay=0.01)
        
        params = np.random.randn(30)
        for _ in range(5):
            grads = np.random.randn(30) * 0.1
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = AdamW(lr=0.001, weight_decay=0.01)
        opt2.load_state_dict(state)
        
        assert opt2.t == opt1.t
        np.testing.assert_array_almost_equal(opt2.m, opt1.m)
        np.testing.assert_array_almost_equal(opt2.v, opt1.v)
    
    def test_sgd_momentum_state_persistence(self):
        """Test SGDMomentum state persistence."""
        opt1 = SGDMomentum(lr=0.01, beta=0.9)
        
        params = np.random.randn(20)
        for _ in range(8):
            grads = np.random.randn(20) * 0.05
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = SGDMomentum(lr=0.01, beta=0.9)
        opt2.load_state_dict(state)
        
        # Verify velocity preserved
        np.testing.assert_array_almost_equal(opt2.v, opt1.v)
        
        # Verify identical updates
        test_grads = np.random.randn(20) * 0.05
        params1 = opt1.step(params, test_grads)
        params2 = opt2.step(params, test_grads)
        np.testing.assert_array_almost_equal(params1, params2)
    
    def test_sgd_nesterov_state_persistence(self):
        """Test SGDNesterov state persistence."""
        opt1 = SGDNesterov(lr=0.01, beta=0.9)
        
        params = (1.5, 2.5)
        for i in range(5):
            grads = (0.1 * (i + 1), 0.15 * (i + 1))
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = SGDNesterov(lr=0.01, beta=0.9)
        opt2.load_state_dict(state)
        
        assert opt2.v_x == opt1.v_x
        assert opt2.v_y == opt1.v_y
    
    def test_rmsprop_state_persistence(self):
        """Test RMSProp state persistence."""
        opt1 = RMSProp(lr=0.001, decay_rate=0.9)
        
        params = np.random.randn(25)
        for _ in range(6):
            grads = np.random.randn(25) * 0.1
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = RMSProp(lr=0.001, decay_rate=0.9)
        opt2.load_state_dict(state)
        
        # Verify squared gradient accumulator preserved
        np.testing.assert_array_almost_equal(opt2.s, opt1.s)
    
    def test_amsgrad_state_persistence(self):
        """Test AMSGrad state persistence (has vhat_max in addition to m, v)."""
        opt1 = AMSGrad(lr=0.001)
        
        params = np.random.randn(15)
        for _ in range(7):
            grads = np.random.randn(15) * 0.1
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = AMSGrad(lr=0.001)
        opt2.load_state_dict(state)
        
        # AMSGrad has vhat_max in addition to standard m, v
        assert opt2.t == opt1.t
        np.testing.assert_array_almost_equal(opt2.m, opt1.m)
        np.testing.assert_array_almost_equal(opt2.v, opt1.v)
        np.testing.assert_array_almost_equal(opt2.vhat_max, opt1.vhat_max)
    
    def test_radam_state_persistence(self):
        """Test RAdam state persistence."""
        opt1 = RAdam(lr=0.001)
        
        params = np.random.randn(18)
        for _ in range(5):
            grads = np.random.randn(18) * 0.05
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = RAdam(lr=0.001)
        opt2.load_state_dict(state)
        
        assert opt2.t == opt1.t
        assert opt2.rho_inf == opt1.rho_inf
        np.testing.assert_array_almost_equal(opt2.m, opt1.m)
        np.testing.assert_array_almost_equal(opt2.v, opt1.v)
    
    def test_lamb_state_persistence(self):
        """Test LAMB state persistence."""
        opt1 = LAMB(lr=0.001, weight_decay=0.01)
        
        params = np.random.randn(22)
        for _ in range(4):
            grads = np.random.randn(22) * 0.08
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = LAMB(lr=0.001, weight_decay=0.01)
        opt2.load_state_dict(state)
        
        assert opt2.t == opt1.t
        np.testing.assert_array_almost_equal(opt2.m, opt1.m)
        np.testing.assert_array_almost_equal(opt2.v, opt1.v)
    
    def test_adabound_state_persistence(self):
        """Test AdaBound state persistence."""
        opt1 = AdaBound(lr=0.001, final_lr=0.1)
        
        params = np.random.randn(12)
        for _ in range(6):
            grads = np.random.randn(12) * 0.1
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = AdaBound(lr=0.001, final_lr=0.1)
        opt2.load_state_dict(state)
        
        assert opt2.t == opt1.t
        np.testing.assert_array_almost_equal (opt2.m, opt1.m)
        np.testing.assert_array_almost_equal(opt2.v, opt1.v)
    
    def test_sam_wrapper_state_persistence(self):
        """Test SAM wrapper saves base optimizer state recursively."""
        # SAM wrapping SGDMomentum
        opt1 = SAM(lr=0.1, rho=0.05, base_optimizer='SGDMomentum', beta=0.9)
        
        params = np.random.randn(20)
        for _ in range(3):
            grads = np.random.randn(20) * 0.1
            params = opt1.step(params, grads)
        
        # Save state
        state = opt1.state_dict()
        
        # Verify state contains both SAM and base optimizer state
        assert 'base_optimizer' in state
        assert 'rho' in state
        assert isinstance(state['base_optimizer'], dict)
        
        # Base optimizer (SGDMomentum) should have its own state
        if state['base_optimizer']:
            assert 'v' in state['base_optimizer'] or 'v_x' in state['base_optimizer']
        
        # Restore
        opt2 = SAM(lr=0.1, rho=0.05, base_optimizer='SGDMomentum', beta=0.9)
        opt2.load_state_dict(state)
        
        assert opt2.rho == opt1.rho
    
    def test_sam_with_adam_base_state_persistence(self):
        """Test SAM with Adam as base optimizer."""
        opt1 = SAM(lr=0.001, rho=0.05, base_optimizer='Adam')
        
        params = np.random.randn(15)
        for _ in range(5):
            grads = np.random.randn(15) * 0.1
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = SAM(lr=0.001, rho=0.05, base_optimizer='Adam')
        opt2.load_state_dict(state)
        
        # Verify base Adam state is preserved
        base_state1 = opt1.base_opt.state_dict()
        base_state2 = opt2.base_opt.state_dict()
        
        assert base_state1['t'] == base_state2['t']
    
    def test_lookahead_wrapper_state_persistence(self):
        """Test Lookahead wrapper saves slow weights + base optimizer state."""
        base_opt = SGDMomentum(lr=0.01, beta=0.9)
        opt1 = Lookahead(base_opt, k=5, alpha=0.5)
        
        params = np.random.randn(18)
        for i in range(8):
            grads = np.random.randn(18) * 0.1
            params = opt1.step(params, grads)
        
        # Save state
        state = opt1.state_dict()
        
        # Verify state contains slow weights and base optimizer
        assert 'step_count' in state
        assert 'k' in state
        assert 'alpha' in state
        assert 'base_optimizer' in state
        
        # Create new optimizer and restore
        base_opt2 = SGDMomentum(lr=0.01, beta=0.9)
        opt2 = Lookahead(base_opt2, k=5, alpha=0.5)
        opt2.load_state_dict(state)
        
        assert opt2.step_count == opt1.step_count
        assert opt2.k == opt1.k
        assert opt2.alpha == opt1.alpha
        
        # Verify slow weights preserved
        if opt1.slow_params is not None and opt2.slow_params is not None:
            np.testing.assert_array_almost_equal(opt2.slow_params, opt1.slow_params)
    
    def test_state_dict_invalid_input(self):
        """Test load_state_dict handles invalid inputs gracefully."""
        opt = Adam(lr=0.001)
        
        # Should raise TypeError for non-dict input
        with pytest.raises(TypeError):
            opt.load_state_dict("not a dict")
        
        with pytest.raises(TypeError):
            opt.load_state_dict(None)
        
        with pytest.raises(TypeError):
            opt.load_state_dict([1, 2, 3])
    
    def test_state_dict_dtype_preservation(self):
        """Test that state_dict preserves numpy dtype (float32)."""
        opt1 = Adam(lr=0.001)
        
        params = np.random.randn(10).astype(np.float32)
        for _ in range(3):
            grads = np.random.randn(10).astype(np.float32) * 0.1
            params = opt1.step(params, grads)
        
        state = opt1.state_dict()
        opt2 = Adam(lr=0.001)
        opt2.load_state_dict(state)
        
        # Verify dtype is float32
        if opt2.m is not None:
            assert opt2.m.dtype == np.float32
        if opt2.v is not None:
            assert opt2.v.dtype == np.float32


    def test_state_persistence_none_handling(self):
        """Test that None state (uninitialized arrays) is handled correctly."""
        opt1 = SGDMomentum(lr=0.01, beta=0.9)
        
        # Get state BEFORE any step (v should be None for arrays)
        state = opt1.state_dict()
        assert state['v'] is None
        
        # Load into new optimizer
        opt2 = SGDMomentum(lr=0.01, beta=0.9)
        opt2.load_state_dict(state)
        assert opt2.v is None
        
        # After loading, should still work correctly
        params = np.random.randn(10)
        grads = np.random.randn(10) * 0.1
        result = opt2.step(params, grads)
        assert result is not None
    
    def test_cross_optimizer_state_isolation(self):
        """Test that different optimizer instances have isolated state."""
        opt1 = Adam(lr=0.001)
        opt2 = Adam(lr=0.001)
        
        params = np.random.randn(10)
        grads1 = np.random.randn(10) * 0.1
        grads2 = np.random.randn(10) * 0.2
        
        # Run different gradients on each
        result1 = opt1.step(params, grads1)
        result2 = opt2.step(params, grads2)
        
        # States should differ
        assert opt1.t == 1
        assert opt2.t == 1
        
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_almost_equal(opt1.m, opt2.m)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
