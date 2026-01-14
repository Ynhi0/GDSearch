"""I/O utility helpers for cross-version compatibility (torch.load/save wrappers)

Provides:
- torch_load_safe: handles `weights_only` fallback for older PyTorch versions
- torch_save_safe: handles `_use_new_zipfile_serialization` fallback
"""
import logging
import torch
from typing import Any


def torch_load_safe(path_or_file: Any, map_location=None, weights_only=None):
    """Safely load a Torch checkpoint handling versions that do not accept
    the `weights_only` parameter. Accepts either a path or a file-like object.
    """
    try:
        if weights_only is None:
            # Adopt safe future-friendly default: explicitly set weights_only=True so
            # torch.load does not implicitly use pickle for arbitrary objects.
            return torch.load(path_or_file, map_location=map_location, weights_only=True)
        else:
            return torch.load(path_or_file, map_location=map_location, weights_only=weights_only)
    except TypeError:
        logging.debug("torch.load does not support weights_only param on this PyTorch version; retrying without it")
        return torch.load(path_or_file, map_location=map_location)


def torch_save_safe(obj: Any, path_or_file: Any, use_new_zipfile_serialization: bool = True):
    """Save using new zipfile serialization when available and fallback otherwise."""
    try:
        if use_new_zipfile_serialization:
            torch.save(obj, path_or_file, _use_new_zipfile_serialization=True)
        else:
            torch.save(obj, path_or_file)
    except TypeError:
        logging.debug("torch.save does not accept _use_new_zipfile_serialization on this PyTorch version; using default save")
        torch.save(obj, path_or_file)
