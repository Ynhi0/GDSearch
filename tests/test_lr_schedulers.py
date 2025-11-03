"""
Unit tests for learning rate schedulers.

Tests all scheduler implementations for correctness.
"""

import pytest
import numpy as np
import math
from src.core.lr_schedulers import (
    ConstantLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    CosineAnnealingWarmRestarts, LinearWarmupScheduler, PolynomialLR,
    OneCycleLR, get_scheduler
)


class DummyOptimizer:
    """Dummy optimizer for testing."""
    def __init__(self, lr=0.1):
        self.lr = lr


class TestConstantLR:
    """Test ConstantLR scheduler."""
    
    def test_constant_lr(self):
        """Learning rate should remain constant."""
        optimizer = DummyOptimizer(lr=0.1)
        scheduler = ConstantLR(optimizer)
        
        for _ in range(100):
            scheduler.step()
            assert scheduler.current_lr == 0.1, "LR should remain constant"
            assert optimizer.lr == 0.1, "Optimizer LR should remain constant"


class TestStepLR:
    """Test StepLR scheduler."""
    
    def test_step_decay(self):
        """Learning rate should decay at step boundaries."""
        optimizer = DummyOptimizer(lr=1.0)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Epoch 0-9: lr = 1.0
        for epoch in range(10):
            scheduler.step()
            assert abs(optimizer.lr - 1.0) < 1e-6, f"Epoch {epoch}: LR should be 1.0"
        
        # Epoch 10-19: lr = 0.1
        for epoch in range(10, 20):
            scheduler.step()
            assert abs(optimizer.lr - 0.1) < 1e-6, f"Epoch {epoch}: LR should be 0.1"
        
        # Epoch 20-29: lr = 0.01
        for epoch in range(20, 30):
            scheduler.step()
            assert abs(optimizer.lr - 0.01) < 1e-6, f"Epoch {epoch}: LR should be 0.01"
    
    def test_step_custom_gamma(self):
        """Test with custom gamma value."""
        optimizer = DummyOptimizer(lr=1.0)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        
        expected_lrs = [1.0] * 5 + [0.5] * 5 + [0.25] * 5
        
        for epoch, expected_lr in enumerate(expected_lrs):
            scheduler.step()
            assert abs(optimizer.lr - expected_lr) < 1e-6, \
                f"Epoch {epoch}: expected {expected_lr}, got {optimizer.lr}"


class TestMultiStepLR:
    """Test MultiStepLR scheduler."""
    
    def test_multistep_decay(self):
        """Learning rate should decay at specified milestones."""
        optimizer = DummyOptimizer(lr=1.0)
        milestones = [10, 20, 30]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        
        expected = {
            5: 1.0,
            10: 0.1,
            15: 0.1,
            20: 0.01,
            25: 0.01,
            30: 0.001,
            35: 0.001,
        }
        
        for epoch in range(40):
            scheduler.step()
            if epoch in expected:
                assert abs(optimizer.lr - expected[epoch]) < 1e-6, \
                    f"Epoch {epoch}: expected {expected[epoch]}, got {optimizer.lr}"


class TestExponentialLR:
    """Test ExponentialLR scheduler."""
    
    def test_exponential_decay(self):
        """Learning rate should decay exponentially."""
        optimizer = DummyOptimizer(lr=1.0)
        gamma = 0.9
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        
        for epoch in range(10):
            scheduler.step()
            expected_lr = 1.0 * (gamma ** epoch)
            assert abs(optimizer.lr - expected_lr) < 1e-6, \
                f"Epoch {epoch}: expected {expected_lr}, got {optimizer.lr}"


class TestCosineAnnealingLR:
    """Test CosineAnnealingLR scheduler."""
    
    def test_cosine_annealing(self):
        """Learning rate should follow cosine curve."""
        optimizer = DummyOptimizer(lr=1.0)
        T_max = 100
        eta_min = 0.0
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        lrs = []
        for epoch in range(T_max + 1):
            scheduler.step()
            lrs.append(optimizer.lr)
        
        # Check key points
        assert abs(lrs[0] - 1.0) < 1e-6, "Initial LR should be base_lr"
        assert abs(lrs[T_max // 2] - 0.5) < 0.05, "Mid-point LR should be ~0.5"
        assert lrs[T_max] < 0.1, "Final LR should be close to eta_min"
        
        # Check monotonicity (should be decreasing)
        assert all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1)), \
            "LR should be monotonically decreasing"
    
    def test_cosine_with_eta_min(self):
        """Test cosine annealing with non-zero minimum."""
        optimizer = DummyOptimizer(lr=1.0)
        T_max = 50
        eta_min = 0.01
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        for epoch in range(T_max + 10):
            scheduler.step()
        
        # LR should not go below eta_min
        assert optimizer.lr >= eta_min - 1e-6, \
            f"LR {optimizer.lr} should not be below eta_min {eta_min}"


class TestCosineAnnealingWarmRestarts:
    """Test CosineAnnealingWarmRestarts scheduler."""
    
    def test_warm_restarts(self):
        """Learning rate should restart periodically."""
        optimizer = DummyOptimizer(lr=1.0)
        T_0 = 10
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1)
        
        lrs = []
        for epoch in range(30):
            scheduler.step()
            lrs.append(optimizer.lr)
        
        # Check that LR resets at T_0 intervals
        # At epochs 0, 10, 20, LR should be close to base_lr
        for restart_epoch in [0, 10, 20]:
            if restart_epoch < len(lrs):
                assert lrs[restart_epoch] > 0.9, \
                    f"LR at restart epoch {restart_epoch} should be close to base_lr"


class TestLinearWarmupScheduler:
    """Test LinearWarmupScheduler."""
    
    def test_warmup_only(self):
        """Test warmup without wrapped scheduler."""
        optimizer = DummyOptimizer(lr=1.0)
        warmup_epochs = 10
        scheduler = LinearWarmupScheduler(optimizer, warmup_epochs=warmup_epochs,
                                         warmup_start_lr=0.0)
        
        lrs = []
        for epoch in range(20):
            scheduler.step()
            lrs.append(optimizer.lr)
        
        # Check linear increase during warmup
        assert abs(lrs[0] - 0.0) < 1e-6, "Initial LR should be warmup_start_lr"
        assert abs(lrs[warmup_epochs - 1] - 1.0) < 0.15, \
            "LR at end of warmup should be close to base_lr"
        
        # After warmup, should stay at base_lr
        for epoch in range(warmup_epochs, 20):
            assert abs(lrs[epoch] - 1.0) < 1e-6, \
                f"LR after warmup should be base_lr at epoch {epoch}"
    
    def test_warmup_with_cosine(self):
        """Test warmup followed by cosine annealing."""
        optimizer = DummyOptimizer(lr=1.0)
        warmup_epochs = 5
        T_max = 20
        
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        scheduler = LinearWarmupScheduler(optimizer, warmup_epochs=warmup_epochs,
                                         after_scheduler=cosine_scheduler,
                                         warmup_start_lr=0.0)
        
        lrs = []
        for epoch in range(30):
            scheduler.step()
            lrs.append(optimizer.lr)
        
        # Check warmup phase
        assert lrs[0] < 0.3, "Initial LR should be low"
        assert lrs[warmup_epochs - 1] > 0.7, "LR at end of warmup should be high"
        
        # Check that LR decreases after warmup (cosine phase)
        assert lrs[warmup_epochs + 5] < lrs[warmup_epochs], \
            "LR should decrease after warmup due to cosine annealing"


class TestPolynomialLR:
    """Test PolynomialLR scheduler."""
    
    def test_polynomial_decay(self):
        """Learning rate should decay polynomially."""
        optimizer = DummyOptimizer(lr=1.0)
        max_epochs = 100
        power = 2.0
        scheduler = PolynomialLR(optimizer, max_epochs=max_epochs, power=power)
        
        for epoch in range(max_epochs):
            scheduler.step()
            expected_lr = 1.0 * ((1 - epoch / max_epochs) ** power)
            assert abs(optimizer.lr - expected_lr) < 1e-5, \
                f"Epoch {epoch}: expected {expected_lr}, got {optimizer.lr}"
        
        # At max_epochs, LR should be 0
        scheduler.step()
        assert optimizer.lr == 0.0, "LR should be 0 at max_epochs"


class TestOneCycleLR:
    """Test OneCycleLR scheduler."""
    
    def test_onecycle_policy(self):
        """Learning rate should follow one-cycle policy."""
        optimizer = DummyOptimizer(lr=0.1)
        max_lr = 1.0
        total_steps = 100
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
        
        lrs = []
        for step in range(total_steps):
            scheduler.step()
            lrs.append(optimizer.lr)
        
        # Check that LR increases then decreases
        max_idx = lrs.index(max(lrs))
        
        # Max should occur around pct_start (default 0.3)
        assert 20 < max_idx < 40, f"Max LR should occur around 30% of training, got {max_idx}"
        
        # Check increasing phase
        assert all(lrs[i] <= lrs[i+1] for i in range(max_idx)), \
            "LR should increase in first phase"
        
        # Check decreasing phase
        assert all(lrs[i] >= lrs[i+1] for i in range(max_idx, len(lrs)-1)), \
            "LR should decrease in second phase"


class TestGetScheduler:
    """Test scheduler factory function."""
    
    def test_get_scheduler_by_name(self):
        """Should create correct scheduler by name."""
        optimizer = DummyOptimizer(lr=0.1)
        
        scheduler = get_scheduler('constant', optimizer)
        assert isinstance(scheduler, ConstantLR)
        
        scheduler = get_scheduler('step', optimizer, step_size=10)
        assert isinstance(scheduler, StepLR)
        
        scheduler = get_scheduler('cosine', optimizer, T_max=100)
        assert isinstance(scheduler, CosineAnnealingLR)
    
    def test_get_scheduler_invalid_name(self):
        """Should raise error for invalid scheduler name."""
        optimizer = DummyOptimizer(lr=0.1)
        
        with pytest.raises(ValueError):
            get_scheduler('invalid_name', optimizer)


class TestSchedulerStateDict:
    """Test scheduler state saving/loading."""
    
    def test_state_dict(self):
        """Should save and load scheduler state correctly."""
        optimizer = DummyOptimizer(lr=1.0)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Run for some epochs
        for _ in range(15):
            scheduler.step()
        
        # Save state (last_epoch is 14 after 15 steps starting from -1)
        state = scheduler.state_dict()
        assert state['last_epoch'] == 14
        
        # Create new scheduler and load state
        optimizer2 = DummyOptimizer(lr=1.0)
        scheduler2 = StepLR(optimizer2, step_size=10, gamma=0.1)
        scheduler2.load_state_dict(state)
        
        assert scheduler2.last_epoch == 14
        assert abs(scheduler2.current_lr - scheduler.current_lr) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
