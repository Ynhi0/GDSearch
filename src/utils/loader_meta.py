from typing import Any, Optional


def set_loader_name(loader: Any, name: str) -> None:
    """Set a loader's name metadata in a typing-safe way (accepts any object).

    Using object/Any prevents static analyzers from complaining that the
    built-in DataLoader type does not declare arbitrary attributes.
    """
    setattr(loader, 'name', name)


def set_loader_split_type(loader: Any, split: Any) -> None:
    """Set an internal split type marker for a loader.

    For convenience we set both `_split_type` and `split_type` so callers
    that access either attribute will find the value at runtime.
    """
    setattr(loader, '_split_type', split)
    setattr(loader, 'split_type', split)


def set_loader_purpose(loader: Any, purpose: str) -> None:
    """Set a human-readable purpose for a loader (e.g., 'validation', 'training')."""
    setattr(loader, 'purpose', purpose)


def get_loader_name(loader: Any) -> Optional[str]:
    return getattr(loader, 'name', None)


def get_loader_split_type(loader: Any) -> Optional[Any]:
    return getattr(loader, '_split_type', getattr(loader, 'split_type', None))


def get_loader_purpose(loader: Any) -> Optional[str]:
    return getattr(loader, 'purpose', None)
