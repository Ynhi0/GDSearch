"""Numeric helper utilities.

Provide safe coercion helpers to convert potentially array-like or pandas values
into Python floats without raising on unusual inputs (Series, Index, 0-d arrays,
array-like containers, tuples, etc.).
"""
from __future__ import annotations
from typing import Any

import numpy as np
import logging


def safe_to_float(x: Any) -> float:
    """Safely coerce a value or array-like to a Python float.

    Behavior mirrors conservative unwrapping:
    - If x is numeric (int/float/numpy scalar) return float(x)
    - If x is a pandas Series/Index, pick last non-NaN element
    - If x is a container (list/tuple/ndarray), pick first (or last where appropriate)
    - Fall back to float(x) and return NaN on failure
    """
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception as e:
        logging.debug("safe_to_float: numeric isinstance check failed: %s", e, exc_info=True)

    try:
        # Handle PyTorch tensors (scalars and small tensors)
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            try:
                if x.numel() == 0:
                    return float(np.nan)
                if x.numel() == 1:
                    return float(x.item())
                # Non-scalar tensor: convert to numpy and recurse
                return safe_to_float(x.detach().cpu().numpy())
            except Exception as e_inner:
                logging.debug("safe_to_float: error while handling torch tensor content: %s", e_inner, exc_info=True)
                return float(np.nan)
    except Exception as e_outer:
        logging.debug("safe_to_float: torch import/handling check failed: %s", e_outer, exc_info=True)

    try:
        import pandas as _pd  # local import to avoid hard deps for code that doesn't use pandas
        if isinstance(x, (_pd.Series, _pd.Index)):
            s = x.dropna()
            arr = np.asarray(s)
            if arr.size == 0:
                return float(np.nan)
            # take last non-NA element
            val = arr.ravel()[-1]
            return safe_to_float(val)
    except Exception as e:
        logging.debug("safe_to_float: pandas handling failed (pandas may be unavailable or value unusual): %s", e, exc_info=True)

    try:
        if isinstance(x, (tuple, list)):
            if len(x) == 0:
                return float(np.nan)
            return safe_to_float(x[0])
    except Exception as e:
        logging.debug("safe_to_float: container (tuple/list) handling failed: %s", e, exc_info=True)

    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float(np.nan)
        if arr.shape == () or arr.size == 1:
            val = arr.item()
            if isinstance(val, (int, float, np.integer, np.floating, str)):
                try:
                    return float(val)
                except Exception:
                    return float(np.nan)
            if hasattr(val, "__float__"):
                try:
                    return float(val)
                except Exception:
                    return float(np.nan)
            return float(np.nan)
        # Non-scalar arrays: take first element and recurse
        try:
            return safe_to_float(arr.ravel()[0])
        except Exception:
            return float(np.nan)
    except Exception as e:
        logging.debug("safe_to_float: array handling failed: %s", e, exc_info=True)

    try:
        # Fall back to converting the string representation (avoids passing complex objects directly to float())
        return float(str(x))
    except Exception:
        return float(np.nan)


def safe_len(obj: object) -> int:
    """Robustly determine the number of elements in an object.

    Handles Python containers, numpy arrays, and torch tensors, returning 0
    for None or unsupported objects. This helper avoids raising on unusual
    inputs and is safe for use in dataset/loader size computations.
    """
    if obj is None:
        return 0

    # numpy arrays (prefer total element count over first-dimension len)
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return int(obj.size)
    except Exception:
        pass

    # torch tensors
    try:
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            try:
                return int(obj.numel())
            except Exception:
                return 0
    except Exception:
        pass

    # Builtin containers (lists, tuples, dicts, etc.)
    try:
        return int(len(obj))
    except Exception:
        pass

    # Fallback: try iterator consuming (not ideal for generators) - avoid expensive ops
    return 0
