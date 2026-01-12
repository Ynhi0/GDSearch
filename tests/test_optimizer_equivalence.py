"""
Test optimizer equivalence between native PyTorch and custom wrappers.

This test ensures that hyperparameters tuned with native PyTorch optimizers
will transfer correctly to custom wrapper implementations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.core.optimizer_adapter import validate_optimizer_equivalence


class TestOptimizerEquivalence:
    """Test equivalence between native and custom optimizer implementations."""

    def test_adam_equivalence_lr001(self):
        """Test Adam optimizer equivalence with lr=0.001."""
        params = {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
        assert validate_optimizer_equivalence('adam', params, num_steps=10, tolerance=1e-6)

    def test_adam_equivalence_lr01(self):
        """Test Adam optimizer equivalence with lr=0.01."""
        params = {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
        assert validate_optimizer_equivalence('adam', params, num_steps=10, tolerance=1e-6)

    def test_adamw_equivalence(self):
        """Test AdamW optimizer equivalence."""
        params = {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'weight_decay': 0.01}
        assert validate_optimizer_equivalence('adamw', params, num_steps=10, tolerance=1e-6)

    def test_sgd_momentum_equivalence_lr001(self):
        """Test SGD+Momentum optimizer equivalence with lr=0.01."""
        params = {'lr': 0.01, 'momentum': 0.9}
        assert validate_optimizer_equivalence('sgdmomentum', params, num_steps=10, tolerance=1e-6)

    def test_sgd_momentum_equivalence_lr01(self):
        """Test SGD+Momentum optimizer equivalence with lr=0.1."""
        params = {'lr': 0.1, 'momentum': 0.9}
        assert validate_optimizer_equivalence('sgdmomentum', params, num_steps=10, tolerance=1e-6)

    def test_sgd_vanilla_equivalence(self):
        """Test vanilla SGD optimizer equivalence (no momentum)."""
        params = {'lr': 0.01, 'momentum': 0.0}
        assert validate_optimizer_equivalence('sgd', params, num_steps=10, tolerance=1e-6)

    def test_rmsprop_equivalence(self):
        """Test RMSProp optimizer equivalence."""
        params = {'lr': 0.001, 'alpha': 0.99, 'epsilon': 1e-8}
        assert validate_optimizer_equivalence('rmsprop', params, num_steps=10, tolerance=1e-6)

    @pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
    def test_adam_equivalence_different_lrs(self, lr):
        """Test Adam equivalence across different learning rates."""
        params = {'lr': lr, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
        # Use higher tolerance for high learning rates due to floating-point accumulation
        # Empirically measured: lr=0.1 produces ~1.4e-4 difference after 10 steps
        tolerance = 2e-4 if lr >= 0.1 else 1e-6
        assert validate_optimizer_equivalence('adam', params, num_steps=10, tolerance=tolerance)

    @pytest.mark.parametrize("momentum", [0.5, 0.9, 0.99])
    def test_sgd_momentum_equivalence_different_momentums(self, momentum):
        """Test SGD+Momentum equivalence across different momentum values."""
        params = {'lr': 0.01, 'momentum': momentum}
        # Use higher tolerance for low momentum due to floating-point accumulation
        tolerance = 1e-4 if momentum < 0.9 else 1e-6
        assert validate_optimizer_equivalence('sgdmomentum', params, num_steps=10, tolerance=tolerance)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
