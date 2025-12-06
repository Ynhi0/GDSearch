"""
Timeout utilities for long-running experiments.

Provides signal-based timeout mechanisms with graceful handling.
"""

import signal
import functools
import logging
from contextlib import contextmanager


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


@contextmanager
def timeout(seconds, error_message="Operation timed out"):
    """
    Context manager for timing out operations.
    
    Args:
        seconds: Maximum time in seconds
        error_message: Custom error message
        
    Example:
        with timeout(300, "Training timed out"):
            train_model()
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore old handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def timeout_decorator(seconds, error_message="Function call timed out"):
    """
    Decorator to add timeout to a function.
    
    Args:
        seconds: Maximum time in seconds
        error_message: Custom error message
        
    Example:
        @timeout_decorator(300)
        def train_model():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timeout(seconds, error_message):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def run_with_timeout(func, args=None, kwargs=None, timeout_sec=None, default_return=None):
    """
    Run a function with optional timeout.
    
    Args:
        func: Function to call
        args: Positional arguments (tuple)
        kwargs: Keyword arguments (dict)
        timeout_sec: Timeout in seconds (None = no timeout)
        default_return: Value to return on timeout
        
    Returns:
        Result of func or default_return on timeout
    """
    args = args or ()
    kwargs = kwargs or {}
    
    if timeout_sec is None:
        return func(*args, **kwargs)
    
    try:
        with timeout(timeout_sec, f"{func.__name__} timed out after {timeout_sec}s"):
            return func(*args, **kwargs)
    except TimeoutError as e:
        logging.warning(f"Timeout: {e}")
        return default_return


# Platform check for signal support
import sys
if sys.platform == "win32":
    # Windows doesn't support signal.SIGALRM
    logging.warning("Timeout mechanisms using SIGALRM not supported on Windows. Timeouts will be ignored.")
    
    @contextmanager
    def timeout(seconds, error_message="Operation timed out"):
        """Dummy timeout context for Windows."""
        logging.debug(f"Timeout ({seconds}s) ignored on Windows")
        yield
    
    def timeout_decorator(seconds, error_message="Function call timed out"):
        """Dummy timeout decorator for Windows."""
        def decorator(func):
            return func
        return decorator
