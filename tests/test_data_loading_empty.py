import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from src.runners.data_loading import validate_dataset_split


class EmptyDataset(Dataset):
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError


def test_validate_dataset_split_empty():
    empty = EmptyDataset()
    loader = DataLoader(empty, batch_size=4)
    res = validate_dataset_split(loader, None, loader)
    assert res['train_size'] == 0
    assert res['test_size'] == 0
    assert res['total_size'] == 0
    assert res['train_ratio'] == 0
    assert res['test_ratio'] == 0
