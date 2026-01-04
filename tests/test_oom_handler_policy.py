import torch
import torch.nn as nn
import pytest
from src.core.oom_handler import oom_safe_train_step


class OOMModelAlways(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)
    def forward(self, x):
        raise RuntimeError('CUDA out of memory')


class OOMModelSizeSensitive(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        # Raise OOM when batch size > 1, succeed on batch size == 1
        if x.size(0) > 1:
            raise RuntimeError('CUDA out of memory')
        # Use a small linear mapping so outputs have grad_fn
        return self.fc(x)

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
    """
    model = OOMModelSizeSensitive()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(2, 10)
    targets = torch.randint(0, 10, (2,))

    loss_value, actual_batch, outputs, tainted = oom_safe_train_step(
        model, optim, criterion, inputs, targets, device=torch.device('cpu'),
        max_retries=3, min_batch_size=1
    )

    assert tainted is True
    assert actual_batch == 1
    assert loss_value >= 0.0
