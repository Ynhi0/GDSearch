import torch
import torch.nn as nn
import pytest
from src.core.oom_handler import oom_safe_train_step


class OOMModelAlways(nn.Module):
    """Model that always raises OOM, uses BatchNorm to test BatchNorm handling."""
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)
    def forward(self, x):
        raise RuntimeError('CUDA out of memory')


class OOMModelSizeSensitive(nn.Module):
    """Model that raises OOM when batch size > 2, succeeds on batch size <= 2.
    
    NOTE: Does NOT use BatchNorm so that batch_size=2 reduction can succeed.
    This tests the OOM recovery mechanism without BatchNorm constraints.
    """
    def __init__(self):
        super().__init__()
        # Use LayerNorm instead of BatchNorm - LayerNorm works with any batch size
        self.ln = nn.LayerNorm(10)
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        # Raise OOM when batch size > 2, succeed on batch size <= 2
        if x.size(0) > 2:
            raise RuntimeError('CUDA out of memory')
        # Use LayerNorm then Linear so outputs have grad_fn
        return self.fc(self.ln(x))

def test_eval_fallback_disabled_raises():
    """Test that OOM with BatchNorm model at min batch size raises error.
    
    GAP #22: The allow_batchnorm_eval_fallback parameter was REMOVED because
    switching to eval() mode during training is scientifically invalid.
    Now the function always raises if batch size becomes too small for BatchNorm.
    """
    model = OOMModelAlways()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(1, 10)
    targets = torch.randint(0, 10, (1,))

    with pytest.raises(RuntimeError, match='Batch size too small for BatchNorm'):
        oom_safe_train_step(model, optim, criterion, inputs, targets, device=torch.device('cpu'),
                            max_retries=1, min_batch_size=1)


def test_oom_recovery_reduces_batch_size():
    """Test that OOM-safe handler successfully reduces batch size.
    
    GAP #22: The eval fallback is no longer supported - only batch size reduction.
    This test verifies that batch size reduction still works correctly.
    
    Uses a model WITHOUT BatchNorm so batch size reduction to 2 can succeed.
    """
    model = OOMModelSizeSensitive()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Start with batch size 4, model will fail until batch <= 2
    inputs = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))

    loss_value, actual_batch, outputs, tainted = oom_safe_train_step(
        model, optim, criterion, inputs, targets, device=torch.device('cpu'),
        max_retries=3, min_batch_size=1
    )

    assert tainted is True
    assert actual_batch == 2  # Should reduce 4 -> 2
    assert loss_value >= 0.0
