from typing import Any, Sized, cast


def len_sized(obj: Any) -> int:
    """Return len(obj) if possible, otherwise try common wrappers.

    - If obj has __len__, return len(obj)
    - If obj has attribute 'dataset' with __len__, return len(obj.dataset)
    - Otherwise raise TypeError

    Using Any and cast avoids static typing complaints when the exact type
    is not known (e.g., torch Dataset or HuggingFace IterableDataset).
    """
    if hasattr(obj, "__len__"):
        # cast to Sized for the type checker and call len
        return len(cast(Sized, obj))
    if hasattr(obj, "dataset") and hasattr(obj.dataset, "__len__"):
        return len(cast(Sized, obj.dataset))
    raise TypeError("Object has no __len__; cannot determine size")
