import pytest
import torch
from torch.utils.data import TensorDataset
from src.core.data_hygiene import DataSplitManager


def test_get_test_loader_raises_before_freeze():
    data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    mgr = DataSplitManager(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=1)

    with pytest.raises(RuntimeError):
        _ = mgr.get_test_loader(batch_size=16)


def test_get_test_loader_after_freeze_allows_access():
    data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    mgr = DataSplitManager(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=1)

    # Freeze hyperparameters and then access test loader
    mgr.freeze_hyperparameters({'lr': 0.01})
    test_loader = mgr.get_test_loader(batch_size=16)
    assert test_loader is not None
