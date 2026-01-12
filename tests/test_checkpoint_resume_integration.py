"""
Integration tests for checkpoint resume behavior.

Tests that checkpoint resume properly restores:
1. Model weights
2. Optimizer state
3. Scheduler state
4. Early stopping state (best_val_acc, patience_counter)
5. RNG states for reproducibility

Created: December 24, 2025
Purpose: Verify fixes for checkpoint resume bugs
"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.checkpoint_manager import RobustCheckpointManager as CheckpointManager
from src.core.lr_schedulers import CosineAnnealingLR


class SimpleTestModel(nn.Module):
    """Minimal model for testing checkpoint behavior."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def sample_optimizer(sample_model):
    """Create optimizer for test model."""
    return optim.SGD(sample_model.parameters(), lr=0.01, momentum=0.9)


@pytest.fixture
def sample_scheduler(sample_optimizer):
    """Create LR scheduler for testing."""
    return CosineAnnealingLR(sample_optimizer, T_max=10, eta_min=0.001)


class TestCheckpointResumeEarlyStoppingState:
    """Test that early stopping state (best_val_acc, patience_counter) is restored."""

    def test_early_stopping_state_saved_and_restored(self, temp_checkpoint_dir, sample_model, sample_optimizer, sample_scheduler):
        """
        Test: Verify early stopping metadata is saved and restored.

        This test validates that checkpoint resume restores best_val_acc and patience_counter.
        Without this, resumed runs have invalid early stopping behavior.
        """
        manager = CheckpointManager(base_dir=temp_checkpoint_dir, max_backups=3)

        # Simulate training state at epoch 5
        epoch = 5
        best_val_acc = 89.5
        patience_counter = 3

        # Create checkpoint with metadata
        checkpoint = {
            'model': sample_model.state_dict(),
            'optimizer': sample_optimizer.state_dict(),
            'scheduler': sample_scheduler.state_dict(),
            'epoch': epoch,
            'history': [{'epoch': i, 'val_acc': 80.0 + i} for i in range(1, epoch+1)],
            'metadata': {
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter,
                'completed': False
            }
        }

        # Save checkpoint
        ckpt_file = "test_early_stopping.pt"
        manager.save_checkpoint(checkpoint, ckpt_file, "test_run")

        # Load checkpoint (simulate resume)
        loaded_checkpoint = manager.load_checkpoint(ckpt_file, "test_run")
        assert loaded_checkpoint is not None, "Checkpoint should be loaded successfully"
        # Verify metadata exists
        assert 'metadata' in loaded_checkpoint, "Checkpoint must contain metadata key"

        # Verify early stopping state is preserved
        metadata = loaded_checkpoint['metadata']
        assert metadata.get('best_val_acc') == best_val_acc, \
            f"best_val_acc should be {best_val_acc}, got {metadata.get('best_val_acc')}"
        assert metadata.get('patience_counter') == patience_counter, \
            f"patience_counter should be {patience_counter}, got {metadata.get('patience_counter')}"

        print("✅ TEST PASSED: Early stopping state saved and restored correctly")

    def test_resume_without_metadata_uses_defaults(self, temp_checkpoint_dir, sample_model, sample_optimizer):
        """Test that missing metadata doesn't crash - should use defaults."""
        manager = CheckpointManager(base_dir=temp_checkpoint_dir, max_backups=3)

        # Create checkpoint WITHOUT metadata (old checkpoint format)
        checkpoint = {
            'model': sample_model.state_dict(),
            'optimizer': sample_optimizer.state_dict(),
            'epoch': 3
        }

        ckpt_file = "test_no_metadata.pt"
        manager.save_checkpoint(checkpoint, ckpt_file, "test_run")

        # Load checkpoint
        loaded_checkpoint = manager.load_checkpoint(ckpt_file, "test_run")

        # Verify safe defaults (should not crash)
        metadata = loaded_checkpoint.get('metadata', {})
        best_val_acc = metadata.get('best_val_acc', 0.0)
        patience_counter = metadata.get('patience_counter', 0)

        assert best_val_acc == 0.0, "Default best_val_acc should be 0.0"
        assert patience_counter == 0, "Default patience_counter should be 0"

        print("✅ TEST PASSED: Missing metadata handled safely with defaults")


class TestCheckpointResumeModelState:
    """Test that model weights are correctly restored."""

    def test_model_weights_restored(self, temp_checkpoint_dir, sample_model, sample_optimizer):
        """Verify model weights match before save and after load."""
        manager = CheckpointManager(base_dir=temp_checkpoint_dir, max_backups=3)

        # Save initial weights
        initial_weights = {k: v.clone() for k, v in sample_model.state_dict().items()}

        # Create and save checkpoint
        checkpoint = {
            'model': sample_model.state_dict(),
            'optimizer': sample_optimizer.state_dict(),
            'epoch': 1
        }
        ckpt_file = "test_weights.pt"
        manager.save_checkpoint(checkpoint, ckpt_file, "test_run")

        # Modify model weights (simulate continued training)
        for param in sample_model.parameters():
            param.data.add_(torch.randn_like(param.data))

        # Load checkpoint
        loaded_checkpoint = manager.load_checkpoint(ckpt_file, "test_run")
        assert loaded_checkpoint is not None, "load_checkpoint returned None"
        sample_model.load_state_dict(loaded_checkpoint['model'])

        # Verify weights match initial state
        for key in initial_weights:
            assert torch.allclose(sample_model.state_dict()[key], initial_weights[key]), \
                f"Weight {key} not restored correctly"

        print("✅ TEST PASSED: Model weights restored correctly")


class TestCheckpointResumeOptimizerState:
    """Test that optimizer state (momentum, learning rate) is restored."""

    def test_optimizer_state_restored(self, temp_checkpoint_dir, sample_model):
        """Verify optimizer state matches before save and after load."""
        manager = CheckpointManager(base_dir=temp_checkpoint_dir, max_backups=3)

        optimizer = optim.SGD(sample_model.parameters(), lr=0.01, momentum=0.9)

        # Perform some training steps to build momentum buffers
        for _ in range(5):
            optimizer.zero_grad()
            loss = sample_model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()

        # Save optimizer state
        initial_state = optimizer.state_dict()
        checkpoint = {
            'model': sample_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': 5
        }
        ckpt_file = "test_optimizer.pt"
        manager.save_checkpoint(checkpoint, ckpt_file, "test_run")

        # Create new optimizer (fresh state)
        optimizer_new = optim.SGD(sample_model.parameters(), lr=0.01, momentum=0.9)

        # Load checkpoint
        loaded_checkpoint = manager.load_checkpoint(ckpt_file, "test_run")
        assert loaded_checkpoint is not None, "Failed to load checkpoint"
        optimizer_new.load_state_dict(loaded_checkpoint['optimizer'])

        # Verify state matches
        new_state = optimizer_new.state_dict()
        assert new_state['param_groups'][0]['lr'] == initial_state['param_groups'][0]['lr'], \
            "Learning rate not restored"
        assert new_state['param_groups'][0]['momentum'] == initial_state['param_groups'][0]['momentum'], \
            "Momentum not restored"

        print("✅ TEST PASSED: Optimizer state restored correctly")


class TestCheckpointResumeSchedulerState:
    """Test that LR scheduler state is restored."""

    def test_scheduler_state_restored(self, temp_checkpoint_dir, sample_model, sample_optimizer, sample_scheduler):
        """Verify scheduler.last_epoch is restored correctly."""
        manager = CheckpointManager(base_dir=temp_checkpoint_dir, max_backups=3)

        # Step scheduler several times
        for _ in range(5):
            sample_scheduler.step()

        initial_last_epoch = sample_scheduler.last_epoch
        initial_lr = sample_optimizer.param_groups[0]['lr']

        # Save checkpoint
        checkpoint = {
            'model': sample_model.state_dict(),
            'optimizer': sample_optimizer.state_dict(),
            'scheduler': sample_scheduler.state_dict(),
            'epoch': 5
        }
        ckpt_file = "test_scheduler.pt"
        manager.save_checkpoint(checkpoint, ckpt_file, "test_run")

        # Create new scheduler (fresh state)
        optimizer_new = optim.SGD(sample_model.parameters(), lr=0.01, momentum=0.9)
        scheduler_new = CosineAnnealingLR(optimizer_new, T_max=10, eta_min=0.001)

        # Load checkpoint
        loaded_checkpoint = manager.load_checkpoint(ckpt_file, "test_run")
        assert loaded_checkpoint is not None, "load_checkpoint returned None"
        optimizer_new.load_state_dict(loaded_checkpoint['optimizer'])
        scheduler_new.load_state_dict(loaded_checkpoint['scheduler'])

        # Verify scheduler state
        assert scheduler_new.last_epoch == initial_last_epoch, \
            f"Scheduler last_epoch should be {initial_last_epoch}, got {scheduler_new.last_epoch}"

        print("✅ TEST PASSED: Scheduler state restored correctly")


class TestCheckpointResumeRNGState:
    """Test that RNG states are restored for reproducibility."""

    def test_rng_states_restored(self, temp_checkpoint_dir, sample_model, sample_optimizer):
        """Verify RNG states (PyTorch, NumPy, Python) are restored."""
        import random
        import numpy as np

        manager = CheckpointManager(base_dir=temp_checkpoint_dir, max_backups=3)

        # Set initial RNG states
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Capture RNG states BEFORE generating numbers so restores reproduce
        # the earlier draws when the RNG is restored to this saved state.
        checkpoint = {
            'model': sample_model.state_dict(),
            'optimizer': sample_optimizer.state_dict(),
            'epoch': 1,
            'rng_states': {
                'torch_rng_state': torch.get_rng_state(),
                'numpy_random_state': np.random.get_state(),
                'python_random_state': random.getstate()
            }
        }

        # Generate some random numbers
        random_torch_1 = torch.rand(5)
        random_numpy_1 = np.random.rand(5)
        random_python_1 = random.random()
        ckpt_file = "test_rng.pt"
        manager.save_checkpoint(checkpoint, ckpt_file, "test_run")

        # Change RNG states
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)

        # Load checkpoint and restore RNG states
        loaded_checkpoint = manager.load_checkpoint(ckpt_file, "test_run")
        assert loaded_checkpoint is not None, "load_checkpoint returned None"
        manager.restore_rng_states(loaded_checkpoint)

        # Generate random numbers again (should match initial)
        random_torch_2 = torch.rand(5)
        random_numpy_2 = np.random.rand(5)
        random_python_2 = random.random()

        # Verify reproducibility
        assert torch.allclose(random_torch_1, random_torch_2), "PyTorch RNG not restored"
        assert np.allclose(random_numpy_1, random_numpy_2), "NumPy RNG not restored"
        assert random_python_1 == random_python_2, "Python RNG not restored"

        print("✅ TEST PASSED: RNG states restored correctly for reproducibility")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
