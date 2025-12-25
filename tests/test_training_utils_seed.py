import torch
from src.core.training_utils import set_seed


def test_set_seed_enforces_deterministic(monkeypatch):
    # Ensure we start from non-deterministic defaults
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(123, deterministic=True)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_preserves_benchmark_when_not_deterministic(monkeypatch):
    # Start with benchmark enabled
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(123, deterministic=False)

    # Benchmark should remain enabled
    assert torch.backends.cudnn.benchmark is True
