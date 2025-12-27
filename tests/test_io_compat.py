import os
from pathlib import Path
import tempfile

import pytest
import torch

from src.core.io_utils import torch_load_safe, torch_save_safe


def test_torch_load_safe_handles_weights_only(monkeypatch):
    """Simulate a torch.load implementation that does not accept `weights_only`.
    `torch_load_safe` should catch the resulting TypeError and retry without the kwarg.
    """
    def fake_load(path_or_file, map_location=None, **kwargs):
        # If caller supplies `weights_only`, surface a TypeError as older torch would
        if 'weights_only' in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {'ok': True}

    monkeypatch.setattr(torch, 'load', fake_load)

    # Should succeed even when caller passes weights_only kwarg
    res = torch_load_safe('dummy', map_location='cpu', weights_only=False)
    assert res == {'ok': True}

    # And should succeed when weights_only is not provided
    res2 = torch_load_safe('dummy', map_location='cpu')
    assert res2 == {'ok': True}


def test_torch_save_safe_fallback(monkeypatch, tmp_path):
    """Simulate a torch.save implementation that does not accept
    `_use_new_zipfile_serialization`. `torch_save_safe` should catch the
    TypeError and fall back to the default save behavior.
    """
    calls = {'saved': False}

    def fake_save(obj, path_or_file, **kwargs):
        if '_use_new_zipfile_serialization' in kwargs:
            # Simulate older torch that doesn't accept this kwarg
            raise TypeError("unexpected keyword argument '_use_new_zipfile_serialization'")
        # Write a tiny marker to the file so caller can assert the file exists
        p = Path(path_or_file)
        with open(str(p), 'wb') as f:
            f.write(b'0')
        calls['saved'] = True

    monkeypatch.setattr(torch, 'save', fake_save)

    ckpt_path = tmp_path / 'compat_ckpt.pt'
    torch_save_safe({'a': 1}, ckpt_path, use_new_zipfile_serialization=True)

    assert ckpt_path.exists()
    assert calls['saved'] is True
