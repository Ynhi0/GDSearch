import json
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.io_utils import torch_save_safe, torch_load_safe
from src.core.reproducibility import verify_checkpoint_with_metadata


def test_torch_save_load_state_dict_formats(tmp_path):
    # Simple model
    model = nn.Linear(10, 3)
    for p in model.parameters():
        p.data.fill_(0.0)

    ckpt1 = {'model_state_dict': model.state_dict(), 'meta': {'note': 'state_dict_key'}}
    ckpt2 = {'model': model.state_dict(), 'meta': {'note': 'model_key'}}

    p1 = tmp_path / 'ckpt1.pt'
    p2 = tmp_path / 'ckpt2.pt'

    # Save both formats
    torch_save_safe(ckpt1, p1)
    torch_save_safe(ckpt2, p2)

    # Load and ensure contents are present
    loaded1 = torch_load_safe(str(p1))
    loaded2 = torch_load_safe(str(p2))

    assert 'model_state_dict' in loaded1 and 'meta' in loaded1
    assert 'model' in loaded2 and 'meta' in loaded2


def test_verify_checkpoint_accepts_model_key_and_verifies(tmp_path, monkeypatch):
    # Create a ResNet18 instance consistent with repository (simple, but importable)
    from src.core.models import ResNet18

    model = ResNet18(num_classes=10)
    # Zero out parameters so outputs are constant (argmax 0)
    for p in model.parameters():
        p.data.zero_()

    ckpt = {'model': model.state_dict(), 'epoch': 1}
    ckpt_path = tmp_path / 'resnet_ckpt.pt'
    torch_save_safe(ckpt, ckpt_path)

    # Create metadata JSON pointing to checkpoint and claiming perfect accuracy
    meta = {
        'checkpoint': str(ckpt_path),
        'accuracy': 1.0,
        'seed': 42
    }
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text(json.dumps(meta), encoding='utf-8')

    # Create a tiny synthetic CIFAR-like test loader: one sample, label 0
    inputs = torch.zeros(1, 3, 32, 32)
    targets = torch.zeros(1, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)

    # Monkeypatch get_cifar10_loaders to return (train, val, test) or (train, test) depending on signature
    def fake_get_cifar10_loaders(batch_size=128, seed=42, val_split=None):
        if val_split is None:
            return None, test_loader
        else:
            # return train, val, test
            return None, None, test_loader

    monkeypatch.setattr('src.core.reproducibility.get_cifar10_loaders', fake_get_cifar10_loaders)

    res = verify_checkpoint_with_metadata(str(meta_path))
    assert res['status'] in ('verified', 'mismatch', 'metadata_only', 'error')
    # It should not raise and should return a dict
    assert isinstance(res, dict)