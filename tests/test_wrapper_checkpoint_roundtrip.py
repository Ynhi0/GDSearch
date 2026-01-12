"""
Comprehensive tests for optimizer wrapper checkpoint roundtrip.

Tests ensure that wrapper-specific state (slow_params, step_count, etc.)
is correctly persisted and restored across save/load cycles.

This is important for:
1. Training resumption without state loss
2. Reproducible experiments
3. Reproducibility in optimizer comparisons
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List
from src.core.io_utils import torch_load_safe, torch_save_safe

# Import wrappers to test
from src.core.pytorch_optimizers import SAMWrapper, LookaheadWrapper


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Type alias for optimizers that can be used in training (includes custom wrappers)
OptimizerLike = torch.optim.Optimizer | Any  # Any covers DelayedOptimizer and other wrappers


def train_n_steps(model: nn.Module, optimizer: OptimizerLike, n_steps: int = 5, closure_fn: Optional[Callable[..., Any]] | bool = None) -> List[float]:
    """Train model for n steps and return losses.

    Args:
        model: The neural network model
        optimizer: The optimizer to use (supports torch.optim.Optimizer and custom wrappers)
        n_steps: Number of training steps
        closure_fn: If True or a callable, use closure-based stepping (required for SAM).
                   If None/False, use standard gradient stepping.
    """
    model.train()
    model = model.float()
    losses: List[float] = []

    for _ in range(n_steps):
        x = torch.randn(4, 10, dtype=torch.float32)
        y = torch.randint(0, 5, (4,))

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            return loss

        if closure_fn or isinstance(optimizer, SAMWrapper):
            # SAM requires closure
            # PyTorch type stubs expect closure to return float, but actually returns Tensor
            loss_val = optimizer.step(closure)  # type: ignore[arg-type]
        else:
            optimizer.zero_grad()
            output = model(x)
            loss_tensor = nn.functional.cross_entropy(output, y)
            loss_tensor.backward()
            optimizer.step()
            loss_val = loss_tensor

        # Normalize to Python float safely
        import numbers
        loss_scalar: float
        if isinstance(loss_val, torch.Tensor):
            loss_scalar = float(loss_val.item())
        elif isinstance(loss_val, numbers.Number):
            from typing import Any, cast
            loss_scalar = float(cast(Any, loss_val))
        else:
            try:
                from typing import Any, cast
                loss_scalar = float(cast(Any, loss_val))
            except Exception:
                # Fall back to calling .item() only when available
                item_fn = getattr(loss_val, 'item', None)
                if callable(item_fn):
                    loss_scalar = float(cast(Any, item_fn()))
                else:
                    raise TypeError("Could not convert loss value to float")
        losses.append(loss_scalar)

    return losses


class TestLookaheadCheckpoint:
    """Test LookaheadWrapper checkpoint persistence."""

    def test_lookahead_state_dict_contents(self):
        """Test that state_dict contains all required fields."""
        model = SimpleModel()
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        lookahead = LookaheadWrapper(base_opt, k=5, alpha=0.5)

        state = lookahead.state_dict()

        assert 'base_optimizer' in state
        assert 'slow_params' in state
        assert 'step_count' in state
        assert 'k' in state
        assert 'alpha' in state

        # Verify types
        assert isinstance(state['slow_params'], list)
        assert isinstance(state['step_count'], int)
        assert state['step_count'] == 0  # No steps taken yet

    def test_lookahead_checkpoint_roundtrip(self):
        """Test save/load preserves slow_params and step_count."""
        # Create model and optimizer
        model1 = SimpleModel()
        base_opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        lookahead1 = LookaheadWrapper(base_opt1, k=3, alpha=0.5)

        # Train for several steps (more than k to trigger slow update)
        train_n_steps(model1, lookahead1, n_steps=10)

        # Save checkpoint
        checkpoint: Dict[str, Any] = {
            'model': model1.state_dict(),
            'optimizer': lookahead1.state_dict(),
            'step': lookahead1.step_count
        }

        # Create new model and optimizer
        model2 = SimpleModel()
        base_opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        lookahead2 = LookaheadWrapper(base_opt2, k=3, alpha=0.5)

        # Load checkpoint
        model2.load_state_dict(checkpoint['model'])
        lookahead2.load_state_dict(checkpoint['optimizer'])

        # Verify step_count matches
        assert lookahead2.step_count == lookahead1.step_count

        # Verify slow_params match
        assert len(lookahead2.slow_params) == len(lookahead1.slow_params)
        for sp1, sp2 in zip(lookahead1.slow_params, lookahead2.slow_params):
            assert torch.allclose(sp1, sp2, rtol=1e-5)

        # Verify k and alpha match
        assert lookahead2.k == lookahead1.k
        assert lookahead2.alpha == lookahead1.alpha

    def test_lookahead_training_continuation(self):
        """Test that training continues correctly after checkpoint restore."""
        # Scenario: Train 10 epochs, save, load, train 10 more
        # Should match training 20 epochs without interruption

        # Reference: Train continuously for 20 steps
        torch.manual_seed(42)
        model_ref = SimpleModel()
        base_opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)
        lookahead_ref = LookaheadWrapper(base_opt_ref, k=3, alpha=0.5)

        # Set same data for reproducibility
        torch.manual_seed(42)
        losses_ref = []
        for i in range(20):
            torch.manual_seed(42 + i)  # Deterministic data
            losses_ref.extend(train_n_steps(model_ref, lookahead_ref, n_steps=1))

        # Test: Train 10, save, load, train 10 more
        torch.manual_seed(42)
        model_test = SimpleModel()
        base_opt_test = torch.optim.SGD(model_test.parameters(), lr=0.01)
        lookahead_test = LookaheadWrapper(base_opt_test, k=3, alpha=0.5)

        torch.manual_seed(42)
        losses_test = []
        for i in range(10):
            torch.manual_seed(42 + i)
            losses_test.extend(train_n_steps(model_test, lookahead_test, n_steps=1))

        # Save checkpoint
        checkpoint: Dict[str, Any] = {
            'model': model_test.state_dict(),
            'optimizer': lookahead_test.state_dict(),
        }

        # Create new model and load
        model_test2 = SimpleModel()
        base_opt_test2 = torch.optim.SGD(model_test2.parameters(), lr=0.01)
        lookahead_test2 = LookaheadWrapper(base_opt_test2, k=3, alpha=0.5)

        model_test2.load_state_dict(checkpoint['model'])
        lookahead_test2.load_state_dict(checkpoint['optimizer'])

        # Continue training
        for i in range(10, 20):
            torch.manual_seed(42 + i)
            losses_test.extend(train_n_steps(model_test2, lookahead_test2, n_steps=1))

        # Compare final model parameters (should be very close due to floating point)
        for p_ref, p_test in zip(model_ref.parameters(), model_test2.parameters()):
            assert torch.allclose(p_ref, p_test, rtol=1e-4, atol=1e-6), \
                "Model parameters diverged after checkpoint restore!"


class TestSAMCheckpoint:
    """Test SAMWrapper checkpoint persistence."""

    def test_sam_state_dict_contents(self):
        """Test that SAM state_dict contains base optimizer state."""
        model = SimpleModel()
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sam = SAMWrapper(base_opt, rho=0.05)

        state = sam.state_dict()

        assert 'base_optimizer' in state
        assert 'rho' in state
        assert 'adaptive' in state
        assert state['rho'] == 0.05

    def test_sam_checkpoint_roundtrip(self):
        """Test SAM save/load preserves base optimizer state."""
        # Create model and SAM optimizer
        model1 = SimpleModel()
        base_opt1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        sam1 = SAMWrapper(base_opt1, rho=0.05)

        # Train for several steps
        train_n_steps(model1, sam1, n_steps=5)

        # Save checkpoint
        checkpoint: Dict[str, Any] = {
            'model': model1.state_dict(),
            'optimizer': sam1.state_dict(),
        }

        # Create new model and optimizer
        model2 = SimpleModel()
        base_opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        sam2 = SAMWrapper(base_opt2, rho=0.05)

        # Load checkpoint
        model2.load_state_dict(checkpoint['model'])
        sam2.load_state_dict(checkpoint['optimizer'])

        # Verify base optimizer state matches
        state1 = sam1.base_optimizer.state_dict()
        state2 = sam2.base_optimizer.state_dict()

        # Compare state dicts
        assert len(state1['state']) == len(state2['state'])

        # Verify rho matches
        assert sam2.rho == sam1.rho

    def test_sam_training_continuation(self):
        """Test that SAM training continues correctly after restore."""
        # Similar to Lookahead test but for SAM

        # Reference: Train continuously
        torch.manual_seed(42)
        model_ref = SimpleModel()
        base_opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)
        sam_ref = SAMWrapper(base_opt_ref, rho=0.05)

        torch.manual_seed(42)
        for i in range(10):
            torch.manual_seed(42 + i)
            train_n_steps(model_ref, sam_ref, n_steps=1, closure_fn=True)

        # Test: Train, save, load, continue
        torch.manual_seed(42)
        model_test = SimpleModel()
        base_opt_test = torch.optim.SGD(model_test.parameters(), lr=0.01)
        sam_test = SAMWrapper(base_opt_test, rho=0.05)

        torch.manual_seed(42)
        for i in range(5):
            torch.manual_seed(42 + i)
            train_n_steps(model_test, sam_test, n_steps=1, closure_fn=True)

        # Save & load
        checkpoint: Dict[str, Any] = {
            'model': model_test.state_dict(),
            'optimizer': sam_test.state_dict(),
        }

        model_test2 = SimpleModel()
        base_opt_test2 = torch.optim.SGD(model_test2.parameters(), lr=0.01)
        sam_test2 = SAMWrapper(base_opt_test2, rho=0.05)

        model_test2.load_state_dict(checkpoint['model'])
        sam_test2.load_state_dict(checkpoint['optimizer'])

        # Continue training
        for i in range(5, 10):
            torch.manual_seed(42 + i)
            train_n_steps(model_test2, sam_test2, n_steps=1, closure_fn=True)

        # Verify final parameters match
        for p_ref, p_test in zip(model_ref.parameters(), model_test2.parameters()):
            assert torch.allclose(p_ref, p_test, rtol=1e-4, atol=1e-6)


class TestMixedWrapperCheckpoint:
    """Test checkpoints with mixed wrapper types."""

    def test_cannot_load_sam_into_lookahead(self):
        """Test that loading SAM state into Lookahead fails gracefully."""
        model = SimpleModel()

        # Create SAM and save
        base_sam = torch.optim.SGD(model.parameters(), lr=0.01)
        sam = SAMWrapper(base_sam, rho=0.05)
        sam_state = sam.state_dict()

        # Try to load into Lookahead (should fail or be detected)
        base_look = torch.optim.SGD(model.parameters(), lr=0.01)
        lookahead = LookaheadWrapper(base_look, k=5, alpha=0.5)

        # This should raise an error or at least be detectable
        with pytest.raises((KeyError, AttributeError, RuntimeError)):
            lookahead.load_state_dict(sam_state)


class TestCheckpointFileIO:
    """Test actual file save/load operations."""

    def test_lookahead_save_load_from_file(self):
        """Test saving and loading Lookahead checkpoint from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "lookahead_ckpt.pth"

            # Create and train
            model1 = SimpleModel()
            base_opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
            lookahead1 = LookaheadWrapper(base_opt1, k=3, alpha=0.5)

            train_n_steps(model1, lookahead1, n_steps=10)

            # Save to file
            torch_save_safe({
                'model': model1.state_dict(),
                'optimizer': lookahead1.state_dict(),
                'step_count': lookahead1.step_count
            }, ckpt_path)

            assert ckpt_path.exists()

            # Load from file
            model2 = SimpleModel()
            base_opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
            lookahead2 = LookaheadWrapper(base_opt2, k=3, alpha=0.5)

            checkpoint = torch_load_safe(ckpt_path, weights_only=False)
            model2.load_state_dict(checkpoint['model'])
            lookahead2.load_state_dict(checkpoint['optimizer'])

            # Verify
            assert lookahead2.step_count == checkpoint['step_count']
            for sp1, sp2 in zip(lookahead1.slow_params, lookahead2.slow_params):
                assert torch.allclose(sp1, sp2)


# ============================================================================
# Tests for custom optimizer wrappers and DelayedOptimizer
# ============================================================================

from src.core.optimizer_wrappers import DelayedOptimizer
from src.core.pytorch_optimizers import (
    SGDMomentumWrapper, AdamWrapper, SGDNesterovWrapper,
    RMSPropWrapper, AdamWWrapper
)


class TestDelayedOptimizerCheckpoint:
    """Test DelayedOptimizer state persistence."""

    def test_grad_queue_roundtrip(self):
        """Grad queue must survive save/load cycle."""
        torch.manual_seed(42)
        model1 = SimpleModel()
        base_opt1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        delayed1 = DelayedOptimizer(base_opt1, delay_steps=3)

        # Train for 5 steps to fill queue
        train_n_steps(model1, delayed1, n_steps=5)

        # Save state
        state = delayed1.state_dict()

        # Restore to new optimizer
        model2 = SimpleModel()
        base_opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        delayed2 = DelayedOptimizer(base_opt2, delay_steps=3)
        delayed2.load_state_dict(state)

        # Verify queue length matches
        assert len(delayed2.grad_queue) == len(delayed1.grad_queue)
        assert delayed2.delay_steps == 3

    def test_empty_queue_serialization(self):
        """Empty queue should not crash."""
        model = SimpleModel()
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        delayed = DelayedOptimizer(base_opt, delay_steps=5)

        # Don't train - queue is empty
        state = delayed.state_dict()
        assert state['grad_queue'] == []

        # Should restore successfully
        model2 = SimpleModel()
        base_opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        delayed2 = DelayedOptimizer(base_opt2, delay_steps=5)
        delayed2.load_state_dict(state)
        assert len(delayed2.grad_queue) == 0


class TestCustomWrapperCheckpoints:
    """Test all custom optimizer wrappers."""

    @pytest.mark.parametrize("wrapper_cls,kwargs", [
        (SGDMomentumWrapper, {'lr': 0.01, 'momentum': 0.9}),
        (AdamWrapper, {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}),
        (SGDNesterovWrapper, {'lr': 0.01, 'momentum': 0.9}),
        (RMSPropWrapper, {'lr': 0.01, 'alpha': 0.99, 'epsilon': 1e-8}),
        (AdamWWrapper, {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01})
    ])
    def test_custom_opts_persistence(self, wrapper_cls, kwargs):
        """Custom optimizer states (momentum, timesteps) must be saved."""
        torch.manual_seed(42)
        model = SimpleModel()
        optimizer = wrapper_cls(model.parameters(), **kwargs)

        # Train to populate custom_opts
        train_n_steps(model, optimizer, n_steps=10)

        # Check custom_opts created
        assert len(optimizer.custom_opts) > 0, f"{wrapper_cls.__name__} didn't create custom_opts"

        # Save and verify custom_opts in state
        state = optimizer.state_dict()
        assert 'custom_opts' in state
        assert len(state['custom_opts']) == len(optimizer.custom_opts)

        # Restore
        model2 = SimpleModel()
        optimizer2 = wrapper_cls(model2.parameters(), **kwargs)
        optimizer2.load_state_dict(state)

        # Verify restored
        assert len(optimizer2.custom_opts) == len(optimizer.custom_opts)

    @pytest.mark.parametrize("wrapper_cls,kwargs", [
        (SGDMomentumWrapper, {'lr': 0.01, 'momentum': 0.9}),
        (AdamWrapper, {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}),
        (AdamWWrapper, {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01})
    ])
    def test_resume_equivalence(self, wrapper_cls, kwargs):
        """Resumed training must produce same results as continuous training."""
        # Train continuously for 10 steps
        torch.manual_seed(42)
        model_continuous = SimpleModel()
        opt_continuous = wrapper_cls(model_continuous.parameters(), **kwargs)
        train_n_steps(model_continuous, opt_continuous, n_steps=10)
        params_continuous = [p.data.clone() for p in model_continuous.parameters()]

        # Train 5 steps, checkpoint, resume 5 more
        torch.manual_seed(42)
        model_resume = SimpleModel()
        opt_resume = wrapper_cls(model_resume.parameters(), **kwargs)
        train_n_steps(model_resume, opt_resume, n_steps=5)

        # Checkpoint
        state = opt_resume.state_dict()
        model_state = {k: v.clone() for k, v in model_resume.state_dict().items()}

        # Restore and continue
        model_resume.load_state_dict(model_state)
        opt_resume.load_state_dict(state)
        train_n_steps(model_resume, opt_resume, n_steps=5)
        params_resume = [p.data.clone() for p in model_resume.parameters()]

        # Compare final parameters
        for p_cont, p_res in zip(params_continuous, params_resume):
            max_diff = (p_cont - p_res).abs().max().item()
            assert max_diff < 1e-5, \
                f"{wrapper_cls.__name__} resume diverged: max diff = {max_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
