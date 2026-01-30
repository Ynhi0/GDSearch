import os
import tempfile
import torch
from src.core.io_utils import torch_load_safe, torch_save_safe


def test_torch_load_save_roundtrip(tmp_path):
    data = {'a': torch.tensor([1,2,3]), 'value': 42}
    p = tmp_path / 'ckpt.pt'
    # Save with torch_save_safe
    with open(p, 'wb') as f:
        torch_save_safe(data, f, use_new_zipfile_serialization=True)

    # Load with torch_load_safe (weights_only True/False/None)
    loaded1 = torch_load_safe(p, map_location='cpu', weights_only=False)
    assert 'value' in loaded1

    loaded2 = torch_load_safe(p, map_location='cpu', weights_only=True)
    assert 'value' in loaded2

    loaded3 = torch_load_safe(p, map_location='cpu', weights_only=None)
    assert 'value' in loaded3
