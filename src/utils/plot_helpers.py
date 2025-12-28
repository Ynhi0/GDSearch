from typing import Sequence, List
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd


def arr_to_numpy_float(x: ArrayLike) -> np.ndarray:
    """Convert array-like or pandas Series to numpy array of floats safely.

    Handles numpy arrays, pandas Series, pandas ExtensionArray and lists.
    This helps satisfy static type-checkers expecting a plain ndarray for plotting.
    """
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    try:
        arr = np.asarray(x, dtype=float)
        return arr
    except Exception:
        # Fall back to list conversion then numpy
        return np.asarray(list(x), dtype=float)


def labels_to_str_sequence(labels: Sequence[object]) -> List[str]:
    """Convert labels sequence to List[str] for matplotlib xticks/yticks."""
    return [str(l) for l in labels]