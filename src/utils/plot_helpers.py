from typing import Any, Iterable, List, Sequence, cast
from numpy.typing import NDArray
import numpy as np
import pandas as pd


def _safe_float(x: object) -> float:
    """Coerce `x` to float in a way that satisfies static checkers.

    Uses `cast(Any, x)` when calling float() to avoid `ConvertibleToFloat`
    complaints from Pyright while preserving runtime behavior.
    Returns float('nan') when conversion fails.
    """
    try:
        return float(cast(Any, x))
    except Exception:
        return float('nan')


def arr_to_numpy_float(x: Any) -> NDArray[np.float64]:
    """Convert arbitrary input to numpy float64 array safely.

    Accepts numpy arrays, pandas Series, iterables, and scalars and performs
    defensive conversions. Returns an np.ndarray with dtype np.float64.
    This implementation uses runtime checks rather than precise static types
    to avoid spurious type-checker failures at third-party API boundaries.
    """
    # Fast path for pandas Series
    if isinstance(x, pd.Series):
        try:
            return x.to_numpy(dtype=float).astype(np.float64)
        except Exception:
            return np.asarray(x.to_numpy(), dtype=np.float64)  # type: ignore[arg-type]

    # Fast path for numpy arrays or array-like objects (avoid passing unknown object directly to numpy)
    if isinstance(x, (np.ndarray, list, tuple)) or hasattr(x, '__array__'):
        try:
            arr = np.asarray(x, dtype=np.float64)  # type: ignore[arg-type]
            if np.iscomplexobj(arr):
                arr = np.real(arr)
            return arr.astype(np.float64)
        except Exception:
            pass

    # If x is iterable, build list of floats element-wise
    if isinstance(x, Iterable):
        out: List[float] = []
        for v in x:
            out.append(_safe_float(v))
        return np.asarray(out, dtype=np.float64)  # type: ignore[arg-type]

    # Otherwise, try to coerce scalar to float
    scalar = _safe_float(x)
    return np.asarray([scalar], dtype=np.float64)  # type: ignore[arg-type]


def ensure_float_list(values: Sequence[object]) -> List[float]:
    """Convert a sequence of objects into a list[float] safely.

    Non-convertible entries are converted to float('nan'). Use this when
    passing numeric-like sequences to functions that expect lists of floats.
    """
    return [_safe_float(v) for v in values]


def ensure_float_scalar(x: object) -> float:
    """Convert an object to a Python float safely; raises on None."""
    if x is None:
        raise TypeError("Expected numeric scalar, got None")
    v = _safe_float(x)
    if np.isnan(v):
        raise TypeError("Could not convert object to float")
    return v


def labels_to_str_sequence(labels: Sequence[object]) -> List[str]:
    """Convert labels sequence to List[str] for matplotlib xticks/yticks."""
    return [str(l) for l in labels]