"""Type guard helpers to coerce inputs to pandas structures for analysis and plotting.

These helpers are intentionally conservative: they coerce numpy arrays, lists, and ExtensionArray
inputs into pd.DataFrame or pd.Series so downstream code can safely use `.iloc`, `.unique()`,
`.columns`, etc., without confusing static analyzers.
"""
from typing import Iterable, Sequence, Union, Optional
from numpy.typing import ArrayLike
import pandas as pd
import numpy as np


def ensure_dataframe(obj: Union[pd.DataFrame, pd.Series, np.ndarray, ArrayLike, Sequence[object], dict]) -> pd.DataFrame:
    """Coerce various inputs to a DataFrame.

    - If already a DataFrame, return as-is.
    - If a Series, wrap as single-column DataFrame.
    - If a numpy array or list, construct a DataFrame where each column is an array column.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    try:
        # Handles list-like or ndarray
        return pd.DataFrame(obj)
    except Exception:
        # Fallback: wrap object into a 1-row DataFrame
        return pd.DataFrame([obj])


def ensure_series(obj: Union[pd.Series, np.ndarray, Sequence[object], object], name: Optional[str] = None) -> pd.Series:
    """Coerce inputs to a Series."""
    if isinstance(obj, pd.Series):
        if name is not None:
            obj.name = name
        return obj
    try:
        s = pd.Series(obj, name=name)
        return s
    except Exception:
        return pd.Series([obj], name=name)
