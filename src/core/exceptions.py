"""
Custom exception classes for GDSearch.

Provides specific exceptions for better error handling and debugging.
"""


class GDSearchError(Exception):
    """Base exception for all GDSearch errors."""
    pass


class CheckpointRestoreError(GDSearchError):
    """Raised when checkpoint restoration fails critically."""
    pass


class ExperimentTimeoutError(GDSearchError):
    """Raised when experiment exceeds time budget."""
    pass


class InvalidConfigError(GDSearchError):
    """Raised when configuration validation fails."""
    pass


class RNGStateError(GDSearchError):
    """Raised when RNG state restoration fails."""
    pass


class DatasetLoadError(GDSearchError):
    """Raised when dataset loading fails after retries."""
    pass


class HyperparameterError(GDSearchError):
    """Raised when hyperparameter validation fails."""
    pass
